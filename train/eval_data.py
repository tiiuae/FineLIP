import json
import cv2
from PIL import Image
import sys
sys.path.append('../..')
import torch
import torch.utils.data as data
import os
import numpy as np
import clip
from torchvision.datasets import CocoCaptions
import pdb

class docci_dataset(data.Dataset):
    def __init__(self, preprocess=None, filter='test'):
        self.image_root = 'evaluation_data/docci/test_images/'
        self.caption_root = 'evaluation_data/docci/docci_descriptions.jsonlines'
        self.total_image = os.listdir(self.image_root)
        self.total_caption = []
        with open(self.caption_root, 'r') as f:
            for line in f:
                line = json.loads(line)
                if line['split'] == filter:
                    self.total_caption.append(line)
                
        if preprocess is None:
            _, self.preprocess = clip.load("ViT-B/16")
        else:
            self.preprocess = preprocess

    def __len__(self):
        return len(self.total_caption)

    def __getitem__(self, index):
        caption_json = self.total_caption[index]
        image_name = caption_json['image_file']
        image = Image.open(self.image_root + image_name)
        image_tensor = self.preprocess(image)
        caption = caption_json['description']
        
        return image_tensor, caption

class urban_dataset(data.Dataset):
    def __init__(self, preprocess=None):
        self.image_root = 'evaluation_data/Urban1k/image/'
        self.caption_root = 'evaluation_data/Urban1k/caption/'
        self.total_image = os.listdir(self.image_root)
        self.total_caption = os.listdir(self.caption_root)
        if preprocess is None:
            _, self.preprocess = clip.load("ViT-B/16")
        else:
            self.preprocess = preprocess

    def __len__(self):
        return len(self.total_caption)

    def __getitem__(self, index):
        caption_name = self.total_caption[index]
        image_name = self.total_caption[index][:-4] + '.jpg'
        image = Image.open(self.image_root + image_name)
        image_tensor = self.preprocess(image)
        f=open(self.caption_root + caption_name)
        caption = f.readlines()[0]
        
        return image_tensor, caption

class coco_dataset(data.Dataset):
    def __init__(self, preprocess=None):
        self.image_root = 'evaluation_data/coco/val2017'
        self.caption_root = 'evaluation_data/coco/annotations/captions_val2017.json'
        self.coco_zipped  = CocoCaptions(root=self.image_root, annFile=self.caption_root, transform=None)
        if preprocess is None:
            _, self.preprocess = clip.load("ViT-B/16")
        else:
            self.preprocess = preprocess

    def __len__(self):
        return len(self.coco_zipped)

    def __getitem__(self, index):
        org_img, org_caption = self.coco_zipped[index]
        image_tensor = self.preprocess(org_img)
        caption = org_caption[0:5]
        return image_tensor, caption

class flickr_dataset(data.Dataset):
    def __init__(self, preprocess=None):
        self.image_root = 'evaluation_data/flickr30k/Images'
        self.caption_root = 'evaluation_data/flickr30k/results_20130124.token'
        self.zipped_dataset = self._get_list()
        if preprocess is None:
            _, self.preprocess = clip.load("ViT-B/16")
        else:
            self.preprocess = preprocess

    def __len__(self):
        return len(self.zipped_dataset) // 5

    def _get_list(self):
        with open(self.caption_root, 'r') as f:
            dataset_zipped = f.readlines()
        return dataset_zipped

    def __getitem__(self, index):
        data = self.zipped_dataset[index*5:(index+1)*5]
        image_name = data[0].split('\t')[0][:-2]
        caption = [data[i].split('\t')[1] for i in range(5)]
        image_full_path = Image.open(os.path.join(self.image_root,image_name))
        image_tensor = self.preprocess(image_full_path)        
        return image_tensor, caption

if __name__ == "__main__":
    flickrDataset = flickr_dataset()
    pdb.set_trace()