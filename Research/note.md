# TODO

1. Get familiar with PyTorch: https://www.youtube.com/watch?v=9aYuQmMJvjA&t=675s
   - CNN
   - Saving models
   - Concatenating models
2. Train the Segmentation Network
3. Test the Segmentation Network
4. 



# Paper summary: Monocular Real-Time Volumetric Performance Capture

 [textured_capture.pdf](C:\Users\Jamie\Documents\University-Stuffs\Research\textured_capture.pdf) 

- Reconstructs textured **3D human** from each **frame** of a video without multi-view studio setup or pre-captured template
- **Progressive surface localization algorithm** and **mesh-free direct rendering** (2 orders faster than brute-force Marching Cube alg.)
- Adopt the **Online Hard Example Mining** ([OHEM](C:\Users\Jamie\Documents\University-Stuffs\Research\OHEM.pdf), [ex1](https://ranmaosong.github.io/2019/07/20/cv-imbalance-between-easy-and-hard-examples/)) technique to suppress failure of challenging examples

## Performance Capture Methods

- *Cue-based*: uses silhouettes, multi-view correspondences, reflective information - could achieve high quality - required many cameras and controlled illumination
- *Template-based*: Joint/face detection -> pose estimation -> template fitting - could be extended to monocular image - most lacked personalized details such as clothing and hairstyles; [Habermann et al.](https://gvv.mpi-inf.mpg.de/projects/LiveCap/) recovered texture detail through creating a textured 3D template.
- *Deep learning*: FCNN used to infer 3D skeletal joint. Saito et al. combined fully convolutional image features with implicit (surface) functions representation.

## Proposed Method

![image-20200925231713047](C:\Users\Jamie\AppData\Roaming\Typora\typora-user-images\image-20200925231713047.png)

## Implementation

- *Segmentation*:
  - Architecture: **U-Net** with **ResNet-18** backbone.
  - Technique: Reducing initial learning rate = **10.0** by **0.95/epoch** (**Adadelta**) 
  - Trainset: https://drive.google.com/file/d/1jDUddrJlUlv5O_JAdb8qZk45EwtEqf_4/view (LIP+Web)
  - Valset: https://drive.google.com/file/d/1FPqz2P51sbnWo1K2FcowPnZCAGC1-_uY/view (LIP+Web)
  - Testset: https://drive.google.com/file/d/1gPkkqwiXKaPWLIIrF7QfvHHOu0B3zDjB/view

- *PIFu*:

  - Architecture: Modified upon [PIFu](PIFu.pdf). 

    - Switched to **HRNetV2-W18-Small-v2** as **shape ==encoder==** for better quality and speed

    - ###### 6 residual blocks for **color encoder** ([transposed convolution?](https://www.cnblogs.com/shine-lee/p/11559825.html) ***==Don't really understand here==***!) check out [conv_arithmetic.pdf](conv_arithmetic.pdf) !![image-20200930013415963](C:\Users\Jamie\AppData\Roaming\Typora\typora-user-images\image-20200930013415963.png)

  - Techniques:

    - **RMSProp** for shape inference; **Adam** for color inference; learning rate = **1e-3**
    - Batch size = **24**, sampled points = **4096/image**
    - Soft one-hot depth vector (**Soft-Z**)
    - Conditional batch normalization (**CBN**) for reducing channel size of MLP (==***Don't understand***==)
    - Train shape inference for 5 epochs, fix it and train texture (color?) inference for another 5 epochs (***which part exactly? ==Do we train the encoders?==***)

  - Trainset: https://drive.google.com/file/d/1jDUddrJlUlv5O_JAdb8qZk45EwtEqf_4/view

  - Valset: https://drive.google.com/file/d/1FPqz2P51sbnWo1K2FcowPnZCAGC1-_uY/view

  - Testset: https://drive.google.com/file/d/1gPkkqwiXKaPWLIIrF7QfvHHOu0B3zDjB/view





## Segmentation (**U-Net**, **ResNet-18** backbone)

### Preprocessing (`data_builder.py`)

- Color Augmentation  `T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0)`
- Normalization to $[-1,1]$ for three channels
- Random Erasing `T.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3), value=0)`
- Scaled to $256\times256,$ preserves perspective (pad with grey)
- 50% Horizontal Flip

### Architecture

