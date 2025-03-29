# CIFAR-100 Classification with Simplified ResNetV2

## Project Overview

This project implements a simplified version of the ResNetV2 architecture for image classification on the CIFAR-100 dataset. The work is based on the Inception-ResNet-v2 architecture originally proposed by Szegedy et al. in their paper ["Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning"](https://paperswithcode.com/method/inception-resnet-v2).

## Technical Implementation

### Model Architecture

The implemented model is a reduced-complexity variant of ResNetV2 with the following key modifications:

1. Reduced width factor (0.5) compared to standard implementations
2. Simplified root block structure
3. Modified bottleneck architecture with Group Normalization
4. Standardized convolutional layers (StdConv2d)

The architecture maintains the fundamental residual connection approach but with reduced computational complexity suitable for educational purposes.

### Training Protocol

The training process incorporates several modern techniques:

- Mixed-precision training using PyTorch AMP
- Cosine annealing learning rate schedule
- AdamW optimizer with weight decay
- Gradient clipping
- Early stopping based on validation accuracy

## Dataset

The model is trained and evaluated on the CIFAR-100 dataset, which consists of:
- 50,000 training images
- 10,000 test images
- 100 fine-grained classes
- 32×32 color images

## Results

[Results will be reported here after model training completes. This section will include quantitative metrics such as training/validation accuracy, loss curves, and comparative analysis with baseline models.]

## References

1. Szegedy, C., Ioffe, S., Vanhoucke, V., & Alemi, A. (2017). Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning. *AAAI Conference on Artificial Intelligence*.
2. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Identity Mappings in Deep Residual Networks. *European Conference on Computer Vision*.
3. Krizhevsky, A. (2009). Learning Multiple Layers of Features from Tiny Images. *University of Toronto Technical Report*.

## Author

[Your Name]  
[Your University]  
[Department]  
[Date]  

*Note: This implementation represents a simplified educational version of the original architecture, adapted for the CIFAR-100 dataset and computational constraints typical in academic environments.*