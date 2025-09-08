import torch
import torch.distributed as dist
from tqdm import tqdm
import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'model'))

from model import finelip
from loss import loss_select
sys.path.append("..")
from arguments import get_args
from sharegpt4v import share4v_val_dataset, share4v_train_dataset

from torch.utils.data.distributed import DistributedSampler
from scheduler import cosine_lr
import subprocess
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import warnings
import wandb 
import random

warnings.filterwarnings("ignore")

def START_SEED(seed=71):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

def push_to_s3(local_path, s3_path):
    command = f"aws s3 cp {local_path} {s3_path}"
    subprocess.run(command, shell=True)

def setup_distributed(backend="nccl", port=None):
    """Initialize distributed training environment.
    support both slurm and torch.distributed.launch
    see torch.distributed.init_process_group() for more details
    """
    num_gpus = torch.cuda.device_count()

    if "SLURM_JOB_ID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NTASKS"])
        node_list = os.environ["SLURM_NODELIST"]
        addr = subprocess.getoutput(f"scontrol show hostname {node_list} | head -n1")
        # specify master port
        if port is not None:
            os.environ["MASTER_PORT"] = str(port)
        elif "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = "29522"
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = addr
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = str(rank % num_gpus)
        os.environ["RANK"] = str(rank)
    else:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

    dist.init_process_group(
        backend=backend,
        world_size=world_size,
        rank=rank,
    )
    torch.cuda.set_device(device=f'cuda:{rank % num_gpus}')
    return rank % num_gpus

def get_embed_size(vit_variant: str) -> int:
    vit_variant = vit_variant.lower()

    if "bigg" in vit_variant:
        return 1280
    elif "l" in vit_variant:
        return 768
    elif "b" in vit_variant:
        return 512
    else:
        raise ValueError(f"Unknown ViT variant: {vit_variant}")

class CLIP_Clean_Train():
    def __init__(self, args, local_rank=0):
        self.args = args
        self.local_rank = local_rank
        self.exp_name = args.exp_name
        self.base_model = args.base_model
        self.model, _ = finelip.load_from_clip(self.base_model, device='cpu', run_finelip= not self.args.run_baseline)
        args.embed_size = get_embed_size(vit_variant=self.base_model)
        if not self.args.run_baseline: self.model.cross_net.__init__(opt=args)
        self.model.criterion = loss_select(opt=args, loss_type=args.loss_finegrain)
        self.model.train()
        self.model.logit_scale = torch.nn.Parameter(torch.ones([]) * args.log_scale)     
        self.model = self.model.cuda()
        
        self.batch_size = args.global_batch_size // torch.cuda.device_count()
        self.accumulation_steps = 512 // args.global_batch_size
        self.num_epoch = args.epochs
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.warmup_length = args.warmup_length
        self.logdir = f"experiments/{self.exp_name}"
        self.ckptdir = self.logdir + "/ckpt"
        os.makedirs(self.ckptdir, exist_ok=True)

        if self.local_rank == 0:
            hyperparameter_defaults = {
                "weight_decay":args.weight_decay,
                "warmup_length":args.warmup_length,
                "log_scale":args.log_scale,
                "batch_size":self.batch_size,
                "lr":self.lr,
                "num_epoch":self.num_epoch,
            }
            # Report to wandb 
            if args.enable_wandb:
                wandb.tensorboard.patch(root_logdir=self.logdir)
                wandb.init(config=hyperparameter_defaults, project="FineLIP", sync_tensorboard=True, save_code=True, name=self.exp_name)
        self.writer = SummaryWriter(self.logdir)
    
        self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
        if self.args.run_baseline: 
            self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay) # use this for baseline
        else:
            self.optimizer = self.create_optimizer()
        self.scaler = torch.cuda.amp.grad_scaler.GradScaler()        

    def create_optimizer(self):
        finelip_params = []
        for n, p in self.model.named_parameters():
            if not any(nd in n for nd in ["cross_net", "criterion"]):
                finelip_params.append(p)
        param_groups = [
            {'params': finelip_params, 'lr': self.lr},
            {'params': self.model.module.cross_net.parameters(), 'lr': self.args.cross_net_lr}
        ]
        self.optimizer = optim.AdamW(param_groups, weight_decay=self.weight_decay)
        return self.optimizer

    def resume_checkpoint(self, checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        remove = checkpoint_path[-21:-13]
        checkpoint = torch.load(checkpoint_path.replace('.pt', '_other.pt').replace(remove, ''), map_location='cpu')

        self.model.module.load_state_dict(state_dict)
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scaler.load_state_dict(checkpoint['scaler'])
        return checkpoint['epoch']
    
    def save_checkpoint(self, epoch):
        if self.base_model == "ViT-B/16":
            name = 'finelip-B.pt'
        elif self.base_model == "ViT-L/14":
            name = 'finelip-L.pt'
        else:
            name = "finelip-others.pt"

        experiment_name = f'{self.ckptdir}/{self.exp_name}_{self.args.global_batch_size}_epoch_{epoch+1}_{name}'
        state_dict = self.model.module.state_dict()
        other_state_dict = {
            'epoch': epoch + 1,
            'optimizer': self.optimizer.state_dict(),
            'scaler': self.scaler.state_dict()
        }
        torch.save(state_dict, experiment_name)
        torch.save(other_state_dict, experiment_name.replace('.pt', '_other.pt').replace(f'epoch_{epoch+1}', ''))
        if self.args.s3_bucket != None:
            push_to_s3(experiment_name, self.args.s3_bucket)
        print(f"saved model to {experiment_name}")

    def train_epoch(self, dataloader, epoch, start_iter=0):
        running_loss = 0.0
        loss_1, loss_3 = 0.0, 0.0
        num_batches_per_epoch = len(dataloader)
        self.optimizer.zero_grad()
        for i, (images, texts, short_text, img_ids) in enumerate(tqdm(dataloader, disable=(self.local_rank != 0))):
            step = num_batches_per_epoch * epoch + i
            if step < start_iter:
                continue
            img_ids = img_ids.cuda()
            images = images.cuda()
            texts = finelip.tokenize(texts, truncate=True).cuda()
            warmup_alpha = float(i) / num_batches_per_epoch if epoch == self.args.embedding_warmup_epochs else 1.0
            
            loss_1, loss_3 = self.model(images, texts, img_ids, 
                                        warmup_alpha, self.local_rank)

            if torch.isnan(loss_3) or torch.isinf(loss_3):
                print("loss_3 is NaN or Inf")
                loss_3 = torch.zeros([], requires_grad=True, device=images.device)

            loss = (loss_1 + loss_3) / self.accumulation_steps                # Normalize our loss (if averaged)      
            loss.backward()
            
            if (i+1) % self.accumulation_steps == 0:             # Wait for several backward steps
                self.optimizer.step()                       # Now we can do an optimizer step
                self.optimizer.zero_grad()
                self.scheduler(step)
            
            running_loss += loss.item()

            dist.all_reduce(loss)
            loss = loss.item() / torch.distributed.get_world_size()

            if step % 1000 == 0:
                if self.local_rank == 0:
                    print("=====================================")
                    for i, param_group in enumerate(self.optimizer.param_groups):
                        print(f"train lr_{i} step {step}: {param_group['lr']}")
                        self.writer.add_scalar(f"hyper/lr_{i}", param_group['lr'], step)
                    print(f"train logit_scale step {step}: {self.model.module.logit_scale.item()}")
                    self.writer.add_scalar("logit_scale/train", self.model.module.logit_scale.item(), step)
                    print(f"train loss step {step}: {loss}")
                    self.writer.add_scalar("Loss/train", loss, step)
                    print(f"train loss lvl1 step {step}: {loss_1}")
                    self.writer.add_scalar("Loss 1/train", loss_1, step)
                    print(f"train loss lvl3 step {step}: {loss_3}")
                    self.writer.add_scalar("Loss 3/train", loss_3, step)
                    print("=====================================")
                    
                    # with torch.no_grad():
                    #     self.model.eval()
                    #     self.test()
                    #     self.model.train()

        return running_loss / num_batches_per_epoch

    @torch.no_grad()
    def test_epoch(self, dataloader):
        for id, (images, text) in enumerate(tqdm(dataloader, disable=(self.local_rank != 0))):

            images = images.cuda()
            image_features = self.model.module.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            text = finelip.tokenize(text, truncate=True).cuda()
            text_feature = self.model.module.encode_text(text)
            text_feature /= text_feature.norm(dim=-1, keepdim=True)

            # i = 0
            correct = 0
            total = 0

            for i in range(text_feature.shape[0]):
                text = text_feature[i]
                sim = text @ image_features.T
                sim = sim.squeeze()
                correct_i = torch.argmax(sim)

                if i==correct_i:
                    correct = correct + 1
                total = total + 1

        return correct/total
    
    def test(self):
        if self.local_rank == 0:
            self.model.eval()
            testset = share4v_val_dataset()
            testloader = torch.utils.data.DataLoader(testset, batch_size=1000, num_workers=32, pin_memory=True)
            with torch.no_grad():    

                acc = self.test_epoch(testloader)
                print("=====================================")
                print(f"test mean of share4v retrieval: {acc}")
                print("=====================================")

            return
    
    def train(self, resume=False, warmup_length=200):
        trainset = share4v_train_dataset()
        train_sampler = DistributedSampler(dataset=trainset, shuffle=True)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size, sampler=train_sampler, num_workers=32, pin_memory=True)

        lrs = [p["lr"] for p in self.optimizer.param_groups]
        self.scheduler = cosine_lr(self.optimizer, base_lrs=lrs, warmup_length=warmup_length, steps=(self.num_epoch * len(train_loader))/self.accumulation_steps)
        if resume:
            start_epoch, resume_iter = self.resume_checkpoint(self.args.resume_path)
        else:
            start_epoch = 0
        resume_iter = 0
        
        for epoch in range(start_epoch, self.num_epoch):
            loss = self.train_epoch(train_loader, epoch, start_iter=resume_iter)
            print("=====================================")
            print(f"loss: {loss} after training epoch: {epoch+1}")
            print("=====================================")
            if self.local_rank == 0:
                self.save_checkpoint(epoch)

if __name__ == "__main__":
    parser = get_args()
    args = parser.parse_args()
    START_SEED(args.seed)

    local_rank = setup_distributed()
    print("DDP Done")
    if local_rank == 0:
        print(f"args: {args}")
   
    trainer = CLIP_Clean_Train(
        args=args,
        local_rank=local_rank
        )
    trainer.train(resume=(args.resume_path != None))
    torch.distributed.destroy_process_group()