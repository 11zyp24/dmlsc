#  Dirichlet Mixtures for Robust Label Estimation in Test-Agnostic Long-Tailed Recognition
This is the source code for the paper: Dirichlet Mixtures for Robust Label Estimation in Test-Agnostic Long-Tailed Recognition.

## Requirements

* python3.8

```bash
pip install -r requirements.txt
```
### Prepare datasets
Download the dataset [ImageNet](http://image-net.org/index), and Cifar dateset will be downloaded automatically. 

## pretrained model

The pretrained models for CIFAR100_LT and ImageNet are obtained based on the work of [NCL](https://github.com/Bazinga699/NCL). While the pretrained model for CIFAR100_LT is available in this repository, we regret that the ImageNet model could not be uploaded due to its large memory footprint.

## Generate logits and label for dmlsc

To generate logits and label for dmlsc, please modify commands below.

**CIFAR100 im100**

```
python gen_logits_label.py --cfg config/CIFAR/CIFAR100/cifar100_im100_NCL_200epoch.yaml --modelpath path/to/model --distpath path/to/save
```

**ImageNet-LT**

```
python gen_logits_label.py --cfg config/ImageNet-LT/ImageNet_LT_x50.yaml --modelpath path/to/model --distpath path/to/save
```
## Usage

To reproduce the main result in the paper, please run

```bash
# run dmlsc on Cifar100-im100
python dmlsc_cifar100_lt.py --model_dir path/to/model --feat_dir path/to/logits --data_dir path/to/dataset

# run dmlsc on ImageNet-LT
python dmlsc_imagenet_lt.py --model_dir path/to/model --feat_dir path/to/logits --data_dir path/to/dataset

```
## Acknowledgment

We thank the authors for the following repositories for code reference:

**[NCL](https://github.com/Bazinga699/NCL)**, **[label-shift-correction](https://github.com/Stomach-ache/label-shift-correction)**, [**DirMixE**](https://github.com/scongl/DirMixE)
