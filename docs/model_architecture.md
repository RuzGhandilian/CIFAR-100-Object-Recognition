# **Detailed Architecture of Inception-ResNet-v2 for CIFAR-100**

## **Stem Block (Modified for 32×32 Input)**
The stem is significantly simplified compared to the original Inception-ResNet-v2 to accommodate CIFAR-100's smaller 32×32 resolution:

1. **Initial Convolution**:
   - `Conv2d(3, 32, kernel_size=3, stride=1, padding=1)`
   - Uses **Weight Standardization** (normalizes weights before convolution)
   - Followed by **GroupNorm (32 groups) + ReLU**

2. **Progressive Feature Extraction**:
   ```python
   Stem(
       Conv(32→64, kernel=3, stride=1) + GN + ReLU,
       Conv(64→96, kernel=3, stride=2) + GN + ReLU,  # Downsample
       Parallel(
           Conv(96→64, kernel=1) + GN + ReLU,
           Conv(96→64, kernel=1) + GN + ReLU → Conv(64→96, kernel=3, padding=1) + GN + ReLU
       ),
       Concatenate(),
       Conv(192→256, kernel=3, stride=2) + GN + ReLU  # Final downsample
   )
   ```

## **Inception-ResNet Blocks**

### **Inception-ResNet-A (×5)**
Each block contains:
1. **Pre-Activation**:
   - GroupNorm → ReLU → Weight Standardized Conv

2. **Parallel Pathways**:
   - **Branch1**: 1×1 convolution (256→32)
   - **Branch2**: 1×1 → 3×3 (256→32→32)
   - **Branch3**: 1×1 → 3×3 → 3×3 (256→32→48→64)
   
3. **Concatenation + Residual**:
   ```python
   output = concat([branch1, branch2, branch3])  # 32+32+64=128 channels
   output = Conv(128→256, kernel=1)  # Linear projection
   output += input  # Residual connection
   ```

### **Reduction-A**
Transition block between Inception-A and Inception-B:
```python
ReductionA(
    Parallel(
        Conv(256→384, kernel=3, stride=2),  # Pathway 1
        Conv(256→192→224, kernel=1→3) + Conv(224→256, kernel=3, stride=2),  # Pathway 2
        MaxPool2d(kernel=3, stride=2)  # Pathway 3
    ),
    Concatenate()  # 384+256+256=896 output channels
)
```

### **Inception-ResNet-B (×10)**
Wider asymmetric convolutions:
1. **Branch1**: 1×1 (896→128)
2. **Branch2**: 1×1 → [1×7 → 7×1] (896→128→160→192)

```python
output = concat([branch1, branch2])  # 128+192=320
output = Conv(320→896, kernel=1) + input  # Residual
```

### **Reduction-B**
Final spatial compression:
```python
ReductionB(
    Parallel(
        Conv(896→256→384, kernel=1→3, stride=2),
        Conv(896→256→288, kernel=1→3) + Conv(288→320, kernel=3, stride=2),
        MaxPool2d(kernel=3, stride=2)
    ),
    Concatenate()  # 384+320+896=1600 channels
)
```

## **Head Architecture**

1. **Final Feature Processing**:
   - 2× Inception-ResNet-C blocks (1×1 and 1×3→3×1 convolutions)
   - Global Average Pooling (1600→1600)

2. **Classification Layer**:
   - Fully Connected (1600→100) with label smoothing (ε=0.1)

## **Key Modifications from Original Paper**
1. **Normalization**:
   - Replaced BatchNorm with **GroupNorm (32 groups)** + **Weight Standardization**
   - Eliminates batch-size dependency

2. **Stem Simplification**:
   - Removed aggressive early downsampling (original: 299→35×35 in stem)
   - Adapted for 32×32→8×8 feature maps

3. **Block Repetition**:
   - Reduced A/B/C block counts (5/10/5 vs original 5/10/5)
   - Maintained residual scaling factor at 0.17

4. **Pre-Activation**:
   - All blocks use GN→ReLU→Conv ordering
   - Improves gradient flow vs original post-activation

This architecture preserves Inception's multi-scale processing while ResNet's skip connections ease training, achieving 75.7% accuracy on CIFAR-100 with efficient 32×32 adaptation.