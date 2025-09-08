import json
from PIL import Image, UnidentifiedImageError
import clip
import torch.utils.data as data
import io
import s3fs

data4v_root = 'data/'
json_name = 'share-captioner_coco_lcs_sam_1246k_1107.json'
image_root = "s3://dataset-bucket/ShareGPT4V/data/"

class share4v_val_dataset(data.Dataset):
    def __init__(self, preprocess=None):
        self.data4v_root = data4v_root
        self.json_name = json_name
        self.image_root = image_root
        self.total_len = 1000
        with open(data4v_root + json_name, 'r',encoding='utf8')as fp:
            self.json_data = json.load(fp)[:self.total_len]
        if preprocess is None:
            _ , self.preprocess = clip.load("ViT-L/14")
        else:
            self.preprocess = preprocess
    def __len__(self):
        return self.total_len

    def __getitem__(self, index):
        caption = self.json_data[index]['conversations'][1]['value']
        caption = caption.replace("\n", " ")
        image_name = self.image_root + self.json_data[index]['image']
        try:
            image = Image.open(io.BytesIO(s3fs.S3FileSystem().open(image_name).read()))
            image = image.convert('RGB')
        except (OSError, UnidentifiedImageError) as e:
            print(f"Error loading image ({image_name})")
            image = Image.new('RGB', (224, 224), color='black')

        image_tensor = self.preprocess(image)
        return image_tensor, caption


class share4v_train_dataset(data.Dataset):
    def __init__(self):
        self.data4v_root = data4v_root
        self.json_name = json_name
        self.image_root = image_root
        self.total_len = 1000
        with open(data4v_root + json_name, 'r',encoding='utf8')as fp:
            self.json_data = json.load(fp)[self.total_len:]
        _ , self.preprocess = clip.load("ViT-L/14")

    def __len__(self):
        return len(self.json_data)

    def __getitem__(self, index):
        caption = self.json_data[index]['conversations'][1]['value']
        caption = caption.replace("\n", " ")
        

        caption_short = caption.split(". ")[0]
        
        image_name = self.image_root + self.json_data[index]['image']

        try:
            image = Image.open(io.BytesIO(s3fs.S3FileSystem().open(image_name).read()))
            image = image.convert('RGB')
        except (OSError, UnidentifiedImageError) as e:
            print(f"Error loading image ({image_name})")
            image = Image.new('RGB', (224, 224), color = 'black')
        
        image_tensor = self.preprocess(image)
        return image_tensor, caption, caption_short, index


if __name__ == "__main__":
    import pdb
    train_dataset = share4v_train_dataset()
    val_dataset = share4v_train_dataset()
    pdb.set_trace()