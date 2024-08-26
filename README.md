# Efficient Convolutional Neural Networks with Structured Ternary Patterns
Code that implements Convolutional Neural Networks using Structured Ternary Patterns (STeP)

## Overview
High efficiency deep learning models are necessary to facilitate their use in devices with limited resources but also to improve resources required for training. Convolutional neural networks typically exert severe demands on local device resources and this conventionally limits their adoption within mobile and embedded platforms. This paper presents work towards utilizing static convolutional filters generated from the space of local binary patterns and Haar features to design efficient convolutional neural network architectures. These are referred to as Structured Ternary Patterns (STeP) and can be generated during network initialization in a systematic way instead of having learnable weight parameters thus reducing the total weight updates. The ternary values require significantly less storage and with the appropriate low level implementation, can also lead to inference improvements.

## How to run:
```python
 python main.py --sn <model_name> --ds cifar10 --bs 32 --wd 5e-4
```

| Argument Name | Description |
| ------------- | ------------- |
| ---sn | Select model_name model from: [VGG16,VGG16_step,mobilenetv2,mobilenetv2,resnet50,resnet50,efficientnetb0,efficientnetb0_step,stepnet] |
| --ds | Select dataset to use from: [tinyimagenet,imagenet16,cifar100,cifar10] |
| --epochs | Number of epochs for training |
| --lr | Initial learning rate |
| --wd | Weight Decay Value |
| --bs | Batch Size |

* Compared to the original implementation the step_block is optimized further with the introduction of grouped convolutions.

## CITE AS FOLLOWS

Christos Kyrkou, "Toward Efficient Convolutional Neural Networks With Structured Ternary Patterns." *IEEE Transactions on Neural Networks and Learning Systems*, 2024, pp. 1-8. doi:[10.1109/TNNLS.2024.3380827](https://doi.org/10.1109/TNNLS.2024.3380827).

## Other Resources
 [arxiv📜 ](https://arxiv.org/abs/2407.14831)
 [zenodo📜 ](https://zenodo.org/uploads/12784350)

 Relevant Dataset: [ImageNet16](https://zenodo.org/records/8027520)
