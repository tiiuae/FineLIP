import argparse

def get_args():
    parser = argparse.ArgumentParser(description="params")
    parser.add_argument("--exp_name", default="experiment", type=str, help="specify experiment name.")
    parser.add_argument("--run_baseline", action="store_true", help="run baseline")
    parser.add_argument("--seed", default=71, type=int, help="seed")
    parser.add_argument("--accumulation_steps", default=4, type=int, help="accumulation steps")
    parser.add_argument("--epochs", default=6, type=int, help="num_epochs")
    parser.add_argument("--lr", default=1e-6, type=float, help="lr.")
    parser.add_argument("--cross_net_lr", default=2e-4, type=float, help="cross_net_lr.")
    parser.add_argument("--weight_decay", default=1e-2, type=float, help="wd.")
    parser.add_argument("--log_scale", default=4.6052, type=float, help="clip temperature.")
    parser.add_argument("--warmup_length", default=200, type=int, help="warmup_length.")
    parser.add_argument("--base_model", default="ViT-B/16", help="CLIP Base Model")
    parser.add_argument("--enable_wandb", action="store_true", help="enable wandb logging")
    parser.add_argument("--s3_bucket", default=None, type=str, help="s3 bucket path")
    parser.add_argument("--debug", action="store_true", help="debug mode")
    parser.add_argument("--global_batch_size", default=128, type=int, help="global batch size")
    parser.add_argument("--resume_path", default=None, type=str, help="resume path")
    
    parser.add_argument('--embed_size', default=512, type=int, help='Dimensionality of the joint embedding.')
    parser.add_argument('--num_patches', default=196, type=int, help='Number of patches.')
    parser.add_argument('--loss_finegrain', default='vse', type=str, help='the objective function for optimization')
    parser.add_argument('--margin', default=0.2, type=float, help='Rank loss margin.')
    parser.add_argument('--max_violation', action='store_true', help='Use max instead of sum in the rank loss.')
    parser.add_argument('--vse_mean_warmup_epochs', type=int, default=1, help='The number of warmup epochs using mean vse loss')
    parser.add_argument('--embedding_warmup_epochs', type=int, default=0, help='The number of epochs for warming up the embedding layer')
    # cross-modal alignment
    parser.add_argument('--aggr_ratio', default=0.4, type=float, help='the aggr rate for visual token')
    parser.add_argument('--sparse_ratio', default=0.5, type=float, help='the sprase rate for visual token') 
    parser.add_argument('--attention_weight', default=0.8, type=int, help='the weight of attention_map for mask prediction') 
    parser.add_argument('--ratio_weight', default=2.0, type=float, help='if use detach for kt loss')
    return parser