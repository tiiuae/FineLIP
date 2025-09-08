# FineLIP 
This repository is the official implementation of FineLIP

**FineLIP: Extending CLIP’s Reach via Fine-Grained Alignment with Longer
Text Inputs**

## Getting Started

### Environment Setup 

Refer to the `requirements.txt` file to prepare the environment. 

### Model Training

For model training, run:
```shell
bash train.sh
```
Refer to `train/arguments.py` for all the defined variables. 

### Model Evaluation

#### Short & Long Caption Retrieval
To run retrieval on short caption datasets [COCO2017, Flickr30k] and long caption datasets [DOCCI, Urban1K], run the following command after preparing the data: 

```shell
bash test.sh
```
Set the appropriate variables in the bash file. 

## 🙏 Acknowledgments  
This project builds upon the following open-source resources: 

- LongCLIP [https://github.com/beichenzbc/Long-CLIP]
- LAPS [https://github.com/CrossmodalGroup/LAPS]

## Citation
If this project benefits your research, please consider citing our work:
```
@InProceedings{Asokan_2025_CVPR,
    author    = {Asokan, Mothilal and Wu, Kebin and Albreiki, Fatima},
    title     = {FineLIP: Extending CLIP's Reach via Fine-Grained Alignment with Longer Text Inputs},
    booktitle = {Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR)},
    month     = {June},
    year      = {2025},
    pages     = {14495-14504}
}
