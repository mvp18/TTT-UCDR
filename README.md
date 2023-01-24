#### TTT-UCDR: Test-time Training for Universal Cross-Domain Retrieval

[ArXiv preprint](https://arxiv.org/abs/2208.09198)

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

#### Reproducing our results

Check .sh files in `src/algos/SnMpNet` for Rotnet, Jigsaw, and Barlow Twins losses.

#### 🎓 Cite

If this code was helpful for your research, consider citing:

```bibtex
@article{paul2022ttt,
  title={TTT-UCDR: Test-time Training for Universal Cross-Domain Retrieval},
  author={Paul, Soumava and Dutta, Titir and Saha, Aheli and Samanta, Abhishek and Biswas, Soma},
  journal={arXiv preprint arXiv:2208.09198},
  year={2022}
}
```
