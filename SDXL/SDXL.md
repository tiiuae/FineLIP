# Finelip-SDXL
To run Finelip-SDXL, please follow the following step.

### 1. Prepare SDXL Model
Download the pre-trained weights of SDXL-base and SDXL-refiner in the following pages: 
[https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
[https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0)

### 2. Prepare the text encoders
Download the pre-trained Finelip-L and Finelip-bigG respectively. 

[https://huggingface.co/BeichenZhang/LongCLIP-L](https://huggingface.co/BeichenZhang/LongCLIP-L)
[https://huggingface.co/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k](https://huggingface.co/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k)

### 3. Start generating images.
Finally, you can run the `sdxl.py` for generating images.

python sdxl.py \
  --ckpt_path /path/to/checkpoint.pth \
  --enable_finelip \
  --img_dir /path/to/images \
  --dataset dataset_name