import json, os
from glob import glob
import torch.utils.data as data

import sys
sys.path.append('..')
from diffusers import DiffusionPipeline
import torch
from open_clip_long import factory as open_clip
import torch.nn as nn
import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from transformers import (
    CLIPImageProcessor,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
)

from SDXL_pipeline import get_image
from SDXL_img2img import image2image

import argparse
from encode_prompt import initialize
import warnings

warnings.filterwarnings("ignore")

class docci_dataset(data.Dataset):
    def __init__(self, filter='test'):
        self.caption_root = '../data/docci/docci_descriptions.jsonlines'
        self.total_caption = []
        with open(self.caption_root, 'r') as f:
            for line in f:
                line = json.loads(line)
                if line['split'] == filter:
                    self.total_caption.append(line)

    def __len__(self):
        return len(self.total_caption)

    def __getitem__(self, index):
        caption_json = self.total_caption[index]
        image_name = caption_json['image_file']
        caption = caption_json['description']
        
        return image_name, caption

class urban_dataset(data.Dataset):
    def __init__(self):
        self.caption_root = '../data/Urban1k/caption/'
        self.total_caption = sorted(glob(f'{self.caption_root}*.txt'))

    def __len__(self):
        return len(self.total_caption)

    def __getitem__(self, index):
        caption_name = self.total_caption[index]
        image_name = caption_name.split('/')[-1][:-4] + '.jpg'
        caption = open(caption_name).read()
        
        return image_name, caption

parser = argparse.ArgumentParser(description='params')
parser.add_argument('--ckpt_path', default='work/experiments/finelip-L.pt', help="ckpt_path")
parser.add_argument('--enable_finelip', action='store_true', help='enable finelip')
parser.add_argument('--img_dir', default='images/finelip', help='output image directory')
parser.add_argument('--dataset', default='docci', help='dataset to use')
args = parser.parse_args()

initialize(args)
# fix the seed
generator = torch.Generator(device='cuda')
generator.manual_seed(1971)

base = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
)
base.to("cuda")

refiner = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    text_encoder_2=base.text_encoder_2,
    vae=base.vae,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
)
refiner.to("cuda")

# Define how many steps and what % of steps to be run on each experts (80/20) here
n_steps = 40
high_noise_frac = 0.8

if args.dataset == 'docci':
    testset = docci_dataset()
elif args.dataset == 'urban1k':
    testset = urban_dataset()
else:
    raise ValueError("Invalid dataset")
test_loader = data.DataLoader(testset, batch_size=1, shuffle=False)

if not os.path.exists(args.img_dir):
    os.makedirs(args.img_dir)
for i, (image_name, caption) in enumerate(test_loader):
    image_name = image_name[0]
    caption = caption[0]
    image = get_image(
        pipe=base,
        prompt=caption,
        num_inference_steps=n_steps,
        denoising_end=high_noise_frac,
        output_type="latent",
        generator=generator,
    ).images

    image = image2image(
        pipe=refiner,
        prompt=caption,
        num_inference_steps=n_steps,
        denoising_start=high_noise_frac,
        image=image,
        generator=generator,
    ).images[0]

    image_name = f"{args.img_dir}/{image_name}"
    image.save(image_name)

print("Done!")
