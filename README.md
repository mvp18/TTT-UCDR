#### TTT-UCDR

An extension of [Universal Cross-Domain Retrieval: Generalizing across Classes and Domains](http://arxiv.org/abs/2108.08356) | [ICCV 2021](http://iccv2021.thecvf.com/).

#### Requirements and Setup

Python - 3.7.6, PyTorch - 1.1.0, CUDA - 9.0, cuDNN - 7.5.1, NVIDIA Driver Version >= 384.13

```
conda create --name torch11 --file requirements.txt
conda activate torch11
conda install pytorch==1.1.0 torchvision==0.3.0 cudatoolkit=9.0 -c pytorch
conda install -c conda-forge tensorboard
pip install --user future
```

#### Download datasets

Check `downloads` folder for scripts. Change path_dataset in `download_sketchy.sh` and `download_tuberlin.sh`.

#### Pretrained Models

Download from [here](https://drive.google.com/drive/folders/1v-ryaykcviyi7d4IdbtRZ0YuUg9L12_b?usp=sharing).