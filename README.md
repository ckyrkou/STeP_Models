# STeP_Models
Code that implements Convolutional Neural Networks using Structured Ternary Patterns (STeP)

👷UNDER DEVELOPMENT

'''

How to run:
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

# CITE AS FOLLOWS
@ARTICLE{10507028,
  author={Kyrkou, Christos},
  journal={IEEE Transactions on Neural Networks and Learning Systems}, 
  title={Toward Efficient Convolutional Neural Networks With Structured Ternary Patterns}, 
  year={2024},
  volume={},
  number={},
  pages={1-8},
  keywords={Kernel;Training;Quantization (signal);Filters;Neural networks;Convolutional neural networks;Convolutional codes;Convolutional neural networks (ConvNets);deep learning (DL);Haar features;local binary patterns (LBPs);object detection},
  doi={10.1109/TNNLS.2024.3380827}}
