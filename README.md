# **CIFAR-100 Image Classification with Inception-ResNet-v2**  
**Best Validation Accuracy: 74.30%**  

---

## **Project Overview**  
This project implements an **Inception-ResNet-v2 inspired architecture** for **CIFAR-100 image classification**. The model combines:  
- **Inception modules** (multi-scale feature extraction)  
- **Residual connections** (improved gradient flow)  
- **Weight Standardization** (replaces BatchNorm)  
- **Group Normalization** (stable training)  
- **Label Smoothing** (regularization)  
- **Mixed-Precision Training (AMP)**  

Optimized for **CIFAR-100** (32×32 images, 100 classes), the implementation includes **early stopping**, **cosine LR scheduling**, and **AdamW optimization**.

---

## **Model Architecture**  

### **Key Components**  
1. **Inception-ResNet Blocks**  
   - **Stem**: Modified for CIFAR-32 (3×3 conv → GroupNorm → ReLU).  
   - **Inception-A**: Parallel 1×1, 3×3, and 5×5 convolutions with residual shortcuts.  
   - **Inception-B**: Asymmetric convolutions (1×7 → 7×1) for wider receptive fields.  
   - **Inception-C**: Depthwise convolutions for efficiency.  

2. **Residual Adaptations**  
   - Each Inception block includes **skip connections** (ResNet-style).  
   - **Pre-activation structure**: GN → ReLU → Conv.  

3. **Reduction Blocks**  
   - **Reduction-A**: Between Inception-A and Inception-B (stride=2).  
   - **Reduction-B**: Before final pooling (max + avg pooling).  

4. **Head**  
   - **Global Average Pooling** → **Dense (2048 → 100)**.  

### **Inception-ResNet-v2 Modifications**  
✔ **Downscaled for CIFAR-100** (original designed for 299×299 inputs).  
✔ **Replaced BatchNorm with GroupNorm + Weight Standardization** (better for small batches).  
✔ **Simplified stem** (no aggressive downsampling).  

---

## **Training Details**  

### **Hyperparameters**  
| Parameter          | Value                     |  
|--------------------|---------------------------|  
| Batch Size         | 128                       |  
| Epochs             | 1000 (Early Stopping)     |  
| Learning Rate      | 1e-3 (Cosine Annealing)   |  
| Optimizer          | AdamW (Weight Decay=1e-3) |  
| Label Smoothing    | 0.1                       |  
| Early Stopping Patience | 15 epochs                 |  


### **Training Techniques**  
✔ **Mixed-Precision (AMP)** → Faster training with `autocast` + `GradScaler`.  
✔ **Cosine Annealing LR** → Smooth decay.  
✔ **Augmentations** → Random crops + flips.  

---

## **Results**  
- **Best Val Accuracy**: **74.30%**  
- **Training Time**: ~300 epochs (early stopping).  
- **Loss Curve**:  
  ![Loss Curve](./generated_images/loss_function.png)  

---

### **Conclusion**  
This **Inception-ResNet-v2 variant** achieves **74.30% accuracy** on CIFAR-100 by:  
- Combining **Inception multi-scale processing** with **ResNet shortcuts**.  
- Using **GroupNorm + Weight Standardization** for stability.  
- Optimizing training with **AMP + AdamW**.  

---

**Reference**: 
- [Inception-ResNet-v2 Paper](https://arxiv.org/abs/1602.07261) | [Code](https://paperswithcode.com/method/inception-resnet-v2)
- [GroupNorm Paper (Wu & He, 2018)](https://arxiv.org/abs/1803.08494)  
- [Weight Standardization (Qiao et al., 2019)](https://arxiv.org/abs/1903.10520)  
- [Batch Size vs. Generalization (Keskar et al., 2017)](https://arxiv.org/abs/1609.04836)  
