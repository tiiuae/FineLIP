import torch
import torch.nn.functional as F
import math
import torch.nn as nn

from .xttn import mask_xattn_one_text


def is_sqr(n):
    a = int(math.sqrt(n))
    return a * a == n


class TokenSparse(nn.Module):
    def __init__(self, embed_dim=512, sparse_ratio=0.6):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.sparse_ratio = sparse_ratio
    
    def forward(self, tokens, attention_x, attention_y):
        
        B_v, L_v, C = tokens.size()

        # (B_v, L_v)
        score = attention_x + attention_y

        num_keep_token = math.ceil(L_v * self.sparse_ratio)
    
        # select the top-k index, (B_v, L_v)
        score_sort, score_index = torch.sort(score, dim=1, descending=True)
        
        # (B_v, L_v * token_ratio)
        keep_policy = score_index[:, :num_keep_token]

        # (B_v, L_v)
        score_mask = torch.zeros_like(score).scatter(1, keep_policy, 1)
        
        # (B_v, L_v * token_ratio, C)
        select_tokens = torch.gather(tokens, dim=1, index=keep_policy.unsqueeze(-1).expand(-1, -1, C))

        # fusion token
        # (B_v, L_v *  (1 - token_ratio) )
        non_keep_policy = score_index[:, num_keep_token:]

        # (B_v, L_v *  (1 - token_ratio), C )
        non_tokens = torch.gather(tokens, dim=1, index=non_keep_policy.unsqueeze(-1).expand(-1, -1, C))
        
        # (B_v, L_v *  (1 - token_ratio) )
        non_keep_score = score_sort[:, num_keep_token:]
        # through softmax function, (B_v, L_v *  (1 - token_ratio) ) -> (B_v, L_v *  (1 - token_ratio), 1)
        non_keep_score = F.softmax(non_keep_score, dim=1).unsqueeze(-1)

        # get fusion token (B_v, 1, C)
        extra_token = torch.sum(non_tokens * non_keep_score, dim=1, keepdim=True) 

        return select_tokens, extra_token, score_mask
                  

# dim_ratio affect GPU memory
class TokenAggregation(nn.Module):
    def __init__(self, dim=512, keeped_patches=64, dim_ratio=0.2):
        super().__init__()
        
        hidden_dim = int(dim * dim_ratio)

        self.weight = nn.Sequential(
                        nn.LayerNorm(dim),
                        nn.Linear(dim, hidden_dim),
                        nn.GELU(),
                        nn.Linear(hidden_dim, keeped_patches)
                        )
        
        self.scale = nn.Parameter(torch.ones(1, 1, 1))
        
    def forward(self, x, keep_policy=None):

        # (B, N, C) -> (B, N, N_s)
        weight = self.weight(x)

        #  (B, N, N_s) -> (B, N_s, N)
        weight = weight.transpose(2, 1) * self.scale       

        if keep_policy is not None:
            # keep_policy (B, N) -> (B, 1, N)
            keep_policy = keep_policy.unsqueeze(1)
            # increase a large number for mask patches
            weight = weight - (1 - keep_policy) * 1e10

        # learning a set of weight matrices
        weight = F.softmax(weight, dim=2)
        
        # (B, N_s, C)
        # multiply with patch features
        x = torch.bmm(weight, x)
        
        return x
    

## sparse + aggregation
class CrossSparseAggrNet_v2(nn.Module):
    def __init__(self, opt=None):
        super().__init__()

        self.opt = opt
        
        self.hidden_dim = opt.embed_size if hasattr(opt, 'embed_size') else 512  
        self.num_patches = opt.num_patches if hasattr(opt, 'num_patches') else 196

        self.sparse_ratio = opt.sparse_ratio if hasattr(opt, 'sparse_ratio') else 0.5
        self.aggr_ratio = opt.aggr_ratio if hasattr(opt, 'aggr_ratio') else 0.4

        self.attention_weight = opt.attention_weight if hasattr(opt, 'attention_weight') else 0.8
        self.ratio_weight = opt.ratio_weight if hasattr(opt, 'ratio_weight') else 2.0
        
        # the number of aggregated patches
        # self.keeped_patches = int(self.num_patches * self.aggr_ratio) # swap
        self.keeped_patches = int(self.num_patches * self.aggr_ratio * self.sparse_ratio) # aggr & default

        # # sparse network
        # self.sparse_net = TokenSparse(embed_dim=self.hidden_dim, 
        #                             sparse_ratio=self.sparse_ratio,
        #                             )
        # aggregation network
        self.aggr_net= TokenAggregation(dim=self.hidden_dim, 
                                        keeped_patches=self.keeped_patches,
                                        )

        # cap aggregation network
        self.cap_aggr_net = TokenAggregation(dim=self.hidden_dim, 
                                        keeped_patches=self.keeped_patches,
                                        )

    def forward_swap(self, img_embs, cap_embs, cap_lens):
        B_v, L_v, C = img_embs.shape

        self.has_cls_token = False if is_sqr(img_embs.shape[1]) else True

        if self.has_cls_token:
            img_cls_emb = img_embs[:, 0:1, :]
            img_spatial_embs = img_embs[:, 1:, :]
        else:
            img_spatial_embs = img_embs

        aggr_token = self.aggr_net(img_spatial_embs)
        
        # feature normalization
        img_spatial_embs_norm = F.normalize(aggr_token, dim=-1)
        cap_embs_norm = F.normalize(cap_embs, dim=-1)

        # compute self-attention
        with torch.no_grad():
            img_spatial_glo_norm = F.normalize(aggr_token.mean(dim=1, keepdim=True), dim=-1)
            img_spatial_self_attention = (img_spatial_glo_norm * img_spatial_embs_norm).sum(dim=-1)

        improve_sims = []
        score_mask_all = []

        for i in range(len(cap_lens)):
            n_word = cap_lens[i]
            cap_i = cap_embs[i, :n_word, :]
            cap_i_expand = cap_embs_norm[i, :n_word, :].unsqueeze(0).repeat(B_v, 1, 1)

            # compute cross-attention
            with torch.no_grad():
                cap_i_glo = F.normalize(cap_i.mean(0, keepdim=True).unsqueeze(0), dim=-1)
                img_spatial_cap_i_attention = (cap_i_glo * img_spatial_embs_norm).sum(dim=-1)

            # selection
            select_tokens, extra_token, score_mask = self.sparse_net(tokens=aggr_token,
                                                                    attention_x=img_spatial_self_attention,
                                                                    attention_y=img_spatial_cap_i_attention,
                                                                    )
            # add fusion token
            keep_spatial_tokens = torch.cat([select_tokens, extra_token], dim=1)

            # add [cls] token
            if self.has_cls_token:
                select_tokens = torch.cat((img_cls_emb, keep_spatial_tokens), dim=1)
            else:
                select_tokens = keep_spatial_tokens

            # patch normalization
            select_tokens = F.normalize(select_tokens, dim=-1)

            # image-text similarity
            sim_one_text = mask_xattn_one_text(img_embs=select_tokens,
                                               cap_i_expand=cap_i_expand,
                                               )

            improve_sims.append(sim_one_text)
            score_mask_all.append(score_mask)

        improve_sims = torch.cat(improve_sims, dim=1)
        score_mask_all = torch.stack(score_mask_all, dim=0)

        if self.training:
            return improve_sims, score_mask_all
        else:
            return improve_sims

    def forward_aggr_image(self, img_embs, cap_embs, cap_lens):
        B_v, L_v, C = img_embs.shape

        self.has_cls_token = False if is_sqr(img_embs.shape[1]) else True

        if self.has_cls_token:
            img_cls_emb = img_embs[:, 0:1, :]
            img_spatial_embs = img_embs[:, 1:, :]
        else:
            img_spatial_embs = img_embs

        aggr_token = self.aggr_net(img_spatial_embs)
        
        # feature normalization
        cap_embs_norm = F.normalize(cap_embs, dim=-1)

        improve_sims = []
        score_mask_all = []

        for i in range(len(cap_lens)):
            n_word = cap_lens[i]
            # cap_i_expand = cap_embs[i, :n_word, :].unsqueeze(0).repeat(B_v, 1, 1)
            cap_i_expand = cap_embs_norm[i, :n_word, :].unsqueeze(0).repeat(B_v, 1, 1)

            keep_spatial_tokens = aggr_token
        
            # add [cls] token
            if self.has_cls_token:
                select_tokens = torch.cat((img_cls_emb, keep_spatial_tokens), dim=1)
            else:
                select_tokens = keep_spatial_tokens

            # patch normalization
            select_tokens = F.normalize(select_tokens, dim=-1)

            # image-text similarity
            sim_one_text = mask_xattn_one_text(img_embs=select_tokens,
                                               cap_i_expand=cap_i_expand,
                                               )

            improve_sims.append(sim_one_text)

        improve_sims = torch.cat(improve_sims, dim=1)

        if self.training:
            return improve_sims, score_mask_all
        else:
            return improve_sims

    def forward_aggr_text(self, img_embs, cap_embs, cap_lens):
        B_v, L_v, C = img_embs.shape

        improve_sims = []
        score_mask_all = []

        for i in range(len(cap_lens)):
            n_word = cap_lens[i] - 1
            cap_i = cap_embs[i, :n_word, :].unsqueeze(0)

            cap_select_tokens = self.cap_aggr_net(cap_i)

            cap_select_tokens = torch.cat((cap_select_tokens, cap_embs[i, n_word:n_word+1, :].unsqueeze(0)), dim=1)

            cap_select_tokens = F.normalize(cap_select_tokens, dim=-1)

            sim_one_text = mask_xattn_one_text(
                img_embs=img_embs,
                cap_i_expand=cap_select_tokens.repeat(B_v, 1, 1)
            )
            improve_sims.append(sim_one_text)
        improve_sims = torch.cat(improve_sims, dim=1)
        if self.training:
            return improve_sims, score_mask_all
        else:
            return improve_sims

    def forward_dual(self, img_embs, cap_embs, cap_lens):

        B_v, L_v, C = img_embs.shape
    
        # feature normalization
        # (B_v, L_v, C)
        img_embs_norm = F.normalize(img_embs, dim=-1)
        # (B_t, L_t, C)
        cap_embs_norm = F.normalize(cap_embs, dim=-1)

        self.has_cls_token = False if is_sqr(img_embs.shape[1]) else True

        #  whether it exists [cls] token
        if self.has_cls_token:
            # (B_v, 1, C)
            img_cls_emb = img_embs[:, 0:1, :]
            img_cls_emb_norm = img_embs_norm[:, 0:1, :]
            img_spatial_embs = img_embs[:, 1:, :]
            img_spatial_embs_norm = img_embs_norm[:, 1:, :]
        else:
            img_spatial_embs = img_embs
            img_spatial_embs_norm = img_embs_norm

        # compute self-attention 
        with torch.no_grad():
            # (B_v, L_v, C) ->  (B_v, 1, C)
            img_spatial_glo_norm = F.normalize(img_spatial_embs.mean(dim=1, keepdim=True), dim=-1)
            # (B_v, L_v, C) -> (B_v, L_v)
            img_spatial_self_attention = (img_spatial_glo_norm * img_spatial_embs_norm).sum(dim=-1)

        improve_sims = []
        score_mask_all = []

        # Introduce text information
        # process each text separately
        for i in range(len(cap_lens)):

            n_word = cap_lens[i]                  
            # (L_t, C)
            cap_i = cap_embs[i, :n_word, :]

            ## compute cross-attention
            with torch.no_grad():               
                # (L_t, C) -> (1, C) -> (1, 1, C)
                cap_i_glo = F.normalize(cap_i.mean(0, keepdim=True).unsqueeze(0), dim=-1)
                # (B_v, L_v, C) -> (B_v, L_v)
                img_spatial_cap_i_attention = (cap_i_glo * img_spatial_embs_norm).sum(dim=-1)

                # (L_v, C) -> (1, C)
                img_glo = F.normalize(img_spatial_embs[i].mean(0, keepdim=True), dim=-1)
                # (L_t, C) -> (L_t)
                cap_i_img_attention = (img_glo * cap_i).sum(dim=-1)

                ## compute self-attention
                # (L_t, C) -> (1, C)
                cap_i_glo_norm = F.normalize(cap_i.mean(0, keepdim=True), dim=-1)
                # (L_t, C) -> (L_t)
                cap_i_self_attention = (cap_i_glo_norm * cap_i).sum(dim=-1)

            # selection
            select_tokens, extra_token, score_mask = self.sparse_net(tokens=img_spatial_embs, 
                                                                     attention_x=img_spatial_self_attention, 
                                                                    attention_y=img_spatial_cap_i_attention,
                                                                    )

            # caption selection
            cap_select_tokens, cap_extra_token, cap_score_mask = self.sparse_net(tokens=cap_i.unsqueeze(0),
                                                                                attention_x=cap_i_self_attention.unsqueeze(0),
                                                                                attention_y=cap_i_img_attention.unsqueeze(0),
                                                                                )
            
            # aggregation
            aggr_tokens = self.aggr_net(select_tokens)

            # caption aggregation
            cap_select_tokens = self.cap_aggr_net(cap_select_tokens)

            # add fusion token
            keep_spatial_tokens = torch.cat([aggr_tokens, extra_token], dim=1)

            # add [cls] token
            if self.has_cls_token:
                select_tokens = torch.cat((img_cls_emb, keep_spatial_tokens), dim=1)
            else:
                select_tokens = keep_spatial_tokens

            # patch normalization
            select_tokens = F.normalize(select_tokens, dim=-1)
            cap_select_tokens = F.normalize(cap_select_tokens, dim=-1)

            # image-text similarity
            # (B_v, 1)
            sim_one_text = mask_xattn_one_text(img_embs=select_tokens, cap_i_expand=cap_select_tokens.repeat(B_v, 1, 1))
            
            improve_sims.append(sim_one_text)
            score_mask_all.append(score_mask)

        # (B_v, B_t)
        improve_sims = torch.cat(improve_sims, dim=1)
        score_mask_all = torch.stack(score_mask_all, dim=0)

        if self.training:
            return improve_sims, score_mask_all
        else:
            return improve_sims

    def forward_dual_aggr(self, img_embs, cap_embs, cap_lens):
        B_v, L_v, C = img_embs.shape
    
        self.has_cls_token = False if is_sqr(img_embs.shape[1]) else True

        #  whether it exists [cls] token
        if self.has_cls_token:
            # (B_v, 1, C)
            img_cls_emb = img_embs[:, 0:1, :]
            img_spatial_embs = img_embs[:, 1:, :]        
        else:
            img_spatial_embs = img_embs

        # aggregation
        aggr_tokens = self.aggr_net(img_spatial_embs)
        
        improve_sims = []
        score_mask_all = []

        # Introduce text information
        # process each text separately
        for i in range(len(cap_lens)):

            n_word = cap_lens[i] - 1          
            # (L_t, C)
            cap_i = cap_embs[i, :n_word, :].unsqueeze(0)

            # caption aggregation
            cap_select_tokens = self.cap_aggr_net(cap_i)

            # add [cls] token
            cap_select_tokens = torch.cat((cap_select_tokens, cap_embs[i, n_word:n_word+1, :].unsqueeze(0)), dim=1)
            if self.has_cls_token:
                select_tokens = torch.cat((img_cls_emb, aggr_tokens), dim=1)
            else:
                select_tokens = aggr_tokens

            # patch normalization
            select_tokens = F.normalize(select_tokens, dim=-1)
            cap_select_tokens = F.normalize(cap_select_tokens, dim=-1)

            # image-text similarity
            # (B_v, 1)
            sim_one_text = mask_xattn_one_text(
                img_embs=select_tokens, 
                cap_i_expand=cap_select_tokens.repeat(B_v, 1, 1)
            )
            
            improve_sims.append(sim_one_text)

        # (B_v, B_t)
        improve_sims = torch.cat(improve_sims, dim=1)

        if self.training:
            return improve_sims, score_mask_all
        else:
            return improve_sims
    
    def forward_clim(self, img_embs, cap_embs, cap_lens):
        B_v, L_v, C = img_embs.shape

        improve_sims = []
        score_mask_all = []

        for i in range(len(cap_lens)):
            n_word = cap_lens[i]
            cap_i = cap_embs[i, :n_word, :]
            cap_i_expand = cap_i.unsqueeze(0).repeat(B_v, 1, 1)

            # image-text similarity
            sim_one_text = mask_xattn_one_text(img_embs=img_embs,
                                               cap_i_expand=cap_i_expand,
                                               )

            improve_sims.append(sim_one_text)
        
        improve_sims = torch.cat(improve_sims, dim=1)
        if self.training:
            return improve_sims, score_mask_all
        else:
            return improve_sims
    
    def forward(self, img_embs, cap_embs, cap_lens):

        B_v, L_v, C = img_embs.shape
    
        # feature normalization
        # (B_v, L_v, C)
        img_embs_norm = F.normalize(img_embs, dim=-1)
        # (B_t, L_t, C)
        cap_embs_norm = F.normalize(cap_embs, dim=-1)

        self.has_cls_token = False if is_sqr(img_embs.shape[1]) else True

        #  whether it exists [cls] token
        if self.has_cls_token:
            # (B_v, 1, C)
            img_cls_emb = img_embs[:, 0:1, :]
            img_cls_emb_norm = img_embs_norm[:, 0:1, :]
            img_spatial_embs = img_embs[:, 1:, :]
            img_spatial_embs_norm = img_embs_norm[:, 1:, :]
        else:
            img_spatial_embs = img_embs
            img_spatial_embs_norm = img_embs_norm

        # compute self-attention 
        with torch.no_grad():
            # (B_v, L_v, C) ->  (B_v, 1, C)
            img_spatial_glo_norm = F.normalize(img_spatial_embs.mean(dim=1, keepdim=True), dim=-1)
            # (B_v, L_v, C) -> (B_v, L_v)
            img_spatial_self_attention = (img_spatial_glo_norm * img_spatial_embs_norm).sum(dim=-1)

        improve_sims = []
        score_mask_all = []

        # Introduce text information
        # process each text separately
        for i in range(len(cap_lens)):

            n_word = cap_lens[i]                  
            # (L_t, C)
            cap_i = cap_embs[i, :n_word, :]
    
            # (B_v, L_t, C)
            cap_i_expand = cap_embs_norm[i, :n_word, :].unsqueeze(0).repeat(B_v, 1, 1)

            ## compute cross-attention
            with torch.no_grad():               
                # (L_t, C) -> (1, C) -> (1, 1, C)
                cap_i_glo = F.normalize(cap_i.mean(0, keepdim=True).unsqueeze(0), dim=-1)
                # (B_v, L_v, C) -> (B_v, L_v)
                img_spatial_cap_i_attention = (cap_i_glo * img_spatial_embs_norm).sum(dim=-1)

            # selection
            select_tokens, extra_token, score_mask = self.sparse_net(tokens=img_spatial_embs, 
                                                                     attention_x=img_spatial_self_attention, 
                                                                    attention_y=img_spatial_cap_i_attention,
                                                                    )

            # aggregation
            aggr_tokens = self.aggr_net(select_tokens)
            # aggr_tokens = select_tokens

            # add fusion token
            keep_spatial_tokens = torch.cat([aggr_tokens, extra_token], dim=1)

            # add [cls] token
            if self.has_cls_token:
                select_tokens = torch.cat((img_cls_emb, keep_spatial_tokens), dim=1)
            else:
                select_tokens = keep_spatial_tokens

            # patch normalization
            select_tokens = F.normalize(select_tokens, dim=-1)

            # image-text similarity
            # (B_v, 1)
            sim_one_text = mask_xattn_one_text(img_embs=select_tokens, 
                                               cap_i_expand=cap_i_expand, 
                                               )
            
            improve_sims.append(sim_one_text)
            score_mask_all.append(score_mask)

        # (B_v, B_t)
        improve_sims = torch.cat(improve_sims, dim=1)
        score_mask_all = torch.stack(score_mask_all, dim=0)

        if self.training:
            return improve_sims, score_mask_all
        else:
            return improve_sims


if __name__ == '__main__':

    pass