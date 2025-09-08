import torch
from tqdm import tqdm
import sys
import os
from torch.utils.data import DataLoader
from torch.nn.functional import normalize

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'model'))
sys.path.append("..")

from fine_lip.model import finelip

# import clip # for CLIP evaluation
# from open_clip_long import factory as open_clip # for bigG model

from sharegpt4v import share4v_val_dataset
from eval_data import urban_dataset, coco_dataset, flickr_dataset, docci_dataset

import argparse
import numpy as np
import warnings
warnings.filterwarnings("ignore")

class CLIP_Clean_Train():
    def __init__(self, local_rank, args):
        self.local_rank = local_rank
        self.test_data = args.test_data
        self.model, self.preprocess  = finelip.load(args.ckpt_path, device='cpu',run_finelip= args.run_finelip)
        # self.model, self.preprocess = clip.load('ViT-L/14', device='cpu') # for CLIP evaluation
        # self.model, _, self.preprocess = open_clip.create_model_and_transforms(
        #     'ViT-bigG-14',
        #     pretrained="experiments/open_clip_pytorch_model.bin",
        #     text_cfg={'context_length': 77, 'vocab_size': 49408, "width": 1280, "heads": 20, "layers": 32},
        # )
        torch.cuda.set_device(device=f'cuda:{local_rank}')
        self.model = self.model.float().cuda()
        
        self.finegrain = args.finegrain


    @torch.no_grad()
    def test_epoch(self):
        all_image_features, all_text_features = [], []
        self.all_finegrain_image_features, self.all_finegrain_text_features = [], []
        self.lengths = [] 

        for id, (images, text) in enumerate(tqdm(self.testloader)):
            images = images.cuda()
            image_features_full = self.model.encode_image_full(images)
            image_features = image_features_full[:, 0, :]
            image_features /= image_features.norm(dim=-1, keepdim=True)

            batch_text_features, batch_text_features_full, batch_lengths = [], [], []
            for cap_list in text:
                text_token = finelip.tokenize(cap_list, truncate=True).cuda()
                # text_token = clip.tokenize(cap_list, truncate=True).cuda() # for CLIP evaluation
                # openclip_tokenizer = open_clip.get_tokenizer('ViT-bigG-14', context_length=77) # for CLIP-bigG evaluation
                # text_token = openclip_tokenizer(cap_list).cuda() # for CLIP-bigG evaluation
                lengths = [torch.nonzero(text_token[i]).size(0) for i in range(text_token.shape[0])]
                text_features_full = self.model.encode_text_full(text_token) @ self.model.text_projection
                text_features = text_features_full[torch.arange(text_features_full.shape[0]), text_token.argmax(dim=-1)]
                text_features /= text_features.norm(dim=-1, keepdim=True)
                batch_text_features.append(text_features.unsqueeze(0))
                batch_text_features_full.append(text_features_full.unsqueeze(0))
                batch_lengths.append(torch.tensor(lengths).unsqueeze(0))

            batch_text_features = torch.cat(batch_text_features, dim=0)
            batch_text_features_full = torch.cat(batch_text_features_full, dim=0)
            batch_lengths = torch.cat(batch_lengths, dim=0)
            text_features = batch_text_features.permute(1, 0, 2).reshape(-1, batch_text_features.shape[-1])

            if self.finegrain:
                text_features_full = batch_text_features_full.permute(1, 0, 2, 3).reshape(-1, batch_text_features_full.shape[-2], batch_text_features_full.shape[-1])
                lengths = batch_lengths.permute(1, 0).reshape(-1)
                self.all_finegrain_image_features.append(image_features_full)
                self.all_finegrain_text_features.append(text_features_full.cpu())
                self.lengths.extend(lengths)

            all_image_features.append(image_features)
            all_text_features.append(text_features)

        all_image_features = torch.cat(all_image_features, dim=0)
        all_text_features = torch.cat(all_text_features, dim=0)
        self.logits_per_image = (all_image_features @ all_text_features.t()).detach().cpu()
        self.logits_per_text = self.logits_per_image.t()

        if self.test_data in ['share','urban', 'docci']:
            self.ground_truth = torch.arange(len(all_text_features)).view(-1, 1)
            results  = self.get_metrics()
        else:
            results  = self.get_metrics_1v5()            

        return results
    
    def test(self):
        self.model.eval()
        if self.test_data == 'share':
            testset = share4v_val_dataset(self.preprocess)
        elif self.test_data == 'urban':
            testset = urban_dataset(self.preprocess)
        elif self.test_data == 'coco':
            testset = coco_dataset(self.preprocess)
        elif self.test_data == 'flickr':
            testset = flickr_dataset(self.preprocess)
        elif self.test_data == 'docci':
            testset = docci_dataset(self.preprocess)

        self.testloader = DataLoader(testset, batch_size=200, num_workers=8, pin_memory=True) #changed batch size from 1000 to 500 due to OOM 
        with torch.no_grad():    
            metrics = self.test_epoch()
            print("=====================================")
            print(f"test mean of {self.test_data} retrieval")
            for k, v in metrics.items():
                if "@" in k:
                    print(f"{k} {format(v,'.4f')}")
            print("=====================================")

        return

    def save_logits(self, logits, name):
        if not os.path.exists(os.path.dirname(name)):
            os.makedirs(os.path.dirname(name))
        np.save(name, logits.numpy())

    def get_metrics(self):
        metrics = {}

        # self.save_logits(self.logits_per_image, f"{os.path.dirname(args.ckpt_path)}/{args.test_data}/logits_per_image.npy")
        if self.finegrain:
            self.all_finegrain_image_features = torch.cat(self.all_finegrain_image_features, dim=0)
            self.all_finegrain_text_features = torch.cat(self.all_finegrain_text_features, dim=0)

            finegrain_logits_per_image = torch.empty_like(self.logits_per_image)
            finegrain_logits_per_text = torch.empty_like(self.logits_per_text)
            chunk_size = 100
            for i in tqdm(range(0, self.all_finegrain_text_features.shape[0], chunk_size)):
                finegrain_logits_per_image[:, i:i+chunk_size] = self.model.cross_net.forward_dual_aggr(self.all_finegrain_image_features, self.all_finegrain_text_features[i:i+chunk_size].cuda(), self.lengths[i:i+chunk_size]).detach().cpu()

            finegrain_logits_per_text = finegrain_logits_per_image.t()

            alpha = 0.8 #[0.8, 0.2]
            # self.save_logits(finegrain_logits_per_image, f"{os.path.dirname(args.ckpt_path)}/{args.test_data}/finegrain_logits_per_image.npy")
            self.logits_per_image = alpha * self.logits_per_image + (1 - alpha) * finegrain_logits_per_image
            self.logits_per_text = alpha * self.logits_per_text + (1 - alpha) * finegrain_logits_per_text

        logits = {"image_to_text": self.logits_per_image, "text_to_image": self.logits_per_text}

        for name, logit in logits.items():
            ranking = torch.argsort(logit, descending=True)
            preds = torch.where(ranking == self.ground_truth)[1]
            preds = preds.detach().cpu().numpy()
            metrics[f"{name}_mean_rank"] = preds.mean() + 1
            metrics[f"{name}_median_rank"] = np.floor(np.median(preds)) + 1
            for k in [1, 5, 10]:
                metrics[f"{name}_R@{k}"] = np.mean(preds < k)

        return metrics

    def get_metrics_1v5(self):
        metrics = {}

        # self.save_logits(self.logits_per_image, f"{os.path.dirname(args.ckpt_path)}/{args.test_data}/logits_per_image.npy")
        if self.finegrain:
            self.all_finegrain_image_features = torch.cat(self.all_finegrain_image_features, dim=0)
            self.all_finegrain_text_features = torch.cat(self.all_finegrain_text_features, dim=0)

            finegrain_logits_per_image = torch.empty_like(self.logits_per_image)
            finegrain_logits_per_text = torch.empty_like(self.logits_per_text)
            chunk_size = 500
            for i in tqdm(range(0, self.all_finegrain_text_features.shape[0], chunk_size)):
                finegrain_logits_per_image[:, i:i+chunk_size] = self.model.cross_net.forward_dual_aggr(self.all_finegrain_image_features, self.all_finegrain_text_features[i:i+chunk_size].cuda(), self.lengths[i:i+chunk_size]).detach().cpu()
            
            finegrain_logits_per_text = finegrain_logits_per_image.t()

            alpha = 0.8 #[0.8, 0.2]
            # self.save_logits(finegrain_logits_per_image, f"{os.path.dirname(args.ckpt_path)}/{args.test_data}/finegrain_logits_per_image.npy")
            self.logits_per_image = alpha * self.logits_per_image + (1 - alpha) * finegrain_logits_per_image
            self.logits_per_text = alpha * self.logits_per_text + (1 - alpha) * finegrain_logits_per_text

        for k in [1, 5, 10]:
            pred_true = 0
            for i in range(self.logits_per_image.shape[0]):
                pred = self.logits_per_image[i]
                values, topk = pred.topk(k)
                for j in range(5):
                    true_index = 5*i + j
                    if true_index in topk:
                        pred_true = pred_true + 1
                        break
            metrics[f"image_to_text_R@{k}"] = pred_true/self.logits_per_image.shape[0]   

        for k in [1, 5, 10]:
            pred_true = 0            
            for i in range(self.logits_per_text.shape[0]):
                pred = self.logits_per_text[i]
                values, topk = pred.topk(k)
                true_index = i//5
                if true_index in topk:
                    pred_true = pred_true + 1
            metrics[f"text_to_image_R@{k}"] = pred_true/self.logits_per_text.shape[0]                     

        return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='params')
    parser.add_argument('--ckpt_path', default='work/checkpoints/finelip-B.pt', help="ckpt_path")
    parser.add_argument('--test_data',default='urban', help='docci,urban,coco,and flicker')
    parser.add_argument('--run_finelip',action='store_true', help='run finelip model')
    parser.add_argument('--finegrain', action='store_true', help='enable finegrain evaluation')
    args = parser.parse_args()

    trainer = CLIP_Clean_Train(
        local_rank=0,
        args=args
        )
    trainer.test()
