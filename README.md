# MSRSegNet
Efficient Multi-Scale Residual Network for Image Classification and Semantic Segmentation

## Results
* Our segmentation network achieves an mIOU of 81 on the PASCAL VOC dataset while running at about 21 fps on NVIDIA TitanX Pascal.
```
http://host.robots.ox.ac.uk:8080/anonymous/NQRTFB.html
```
* Our segmentation network achieves an mIOU of 61.12 on the PASCAL VOC dataset while running at about 60 fps on NVIDIA TitanX Pascal with an input image of size 224 x 224.
```
http://host.robots.ox.ac.uk:8080/anonymous/IJV89W.html
```
* Our classification network achieves a top-5 error of 8.47% on the ImageNet Validation dataset.


## Datasets
* Download the CamVid dataset from below github repository

```
https://github.com/alexgkendall/SegNet-Tutorial
```
* Download the PASCAL VOC 2012 dataset using below link

```
wget http://cvlab.postech.ac.kr/research/deconvnet/data/VOC2012_SEG_AUG.tar.gz
```

## Training Semantic Segmentation Models
You can train the network as:
* Training M-RiR using GPU-0 on the CamVid dataset

  ```
CUDA_VISIBLE_DEVICES=0 th main.lua --dataset cv --imHeight 384 -imWidth 480 --modelType 2 -lr 0.0001 -d 10 -de 300 -optimizer adam -maxEpoch 100
  ```
  
* Training M-Plain using GPU-0 and GPU-1 on the CAMVID dataset

```
CUDA_VISIBLE_DEVICES=0,1 th main.lua --dataset cv --imHeight 384 -imWidth 480 --modelType 1 -lr 0.0001 -d 10 -de 300 -optimizer adam -maxEpoch 100
```
* Training M-Hyper on the CAMVID dataset

  ```
    CUDA_VISIBLE_DEVICES=1,2 using GPU-1 and GPU-2 th main.lua --dataset cv --imHeight 384 -imWidth 480 --modelType 3 -lr 0.0001 -d 10 -de 300 -optimizer adam -maxEpoch 100
  ```
  
## Training Image Classification Models
* For training on the ImageNet and the Cifar datasets, please use the scripts provided by [FaceBook AI Research](https://github.com/facebook/fb.resnet.torch)
* Once you get the source code from [FaceBook AI Research](https://github.com/facebook/fb.resnet.torch) github repository, please replace the content of resnet.lua by mresnet_class.lua
* Follow the instructions mentioned on FaceBook AI Research Github page for training.
* For doing depth related studies, please follow below command

To train MResNet on Cifar10 dataset with a depth of 11
```
th main.lua -dataset cifar10 -nGPU 2 -batchSize 128 -depth 1
```

To train MResNet on Cifar100 dataset with a depth of 20
```
th main.lua -dataset cifar100 -nGPU 2 -batchSize 128 -depth 2
```


## License
This software is released under a creative commons license which allows for personal and research use only. For a commercial license please contact the authors. You can view a license summary [here](http://creativecommons.org/licenses/by-nc/4.0/).
