import torch
import torch.distributed as dist

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    # if use distributed training
    if not is_dist_avail_and_initialized():
        return tensor

    tensors_gather = [
        torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output

def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]

# come from BLIP, https://github.com/salesforce/BLIP/blob/b7bb1eeb6e901044a9eb1016f408ee908b216bc7/models/blip_retrieval.py#L306
# Gather tensors from all workers with support for backward propagation:
# This implementation does not cut the gradients as torch.distributed.all_gather does.
class GatherLayer(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        output = [torch.zeros_like(x) for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        
        # op=torch.distributed.ReduceOp.SUM
        torch.distributed.all_reduce(all_gradients)

        return all_gradients[torch.distributed.get_rank()]


# Performs all_gather operation on the provided tensors.
# Graph remains connected for backward grad computation.
def all_gather_with_grad(tensors):
    
    # if use distributed training
    if not is_dist_avail_and_initialized():
        return tensors

    tensor_all = GatherLayer.apply(tensors)

    return torch.cat(tensor_all, dim=0)