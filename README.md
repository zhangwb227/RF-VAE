# Title

**## RF-VAE: Reconstruction of Missing Information in Nighttime VIS-NIR Bands Satellite Imagery via Generative AI**

# Contributions

1. A RF-VAE is proposed, consisting of a conditional encoder, a target encoder, and a target decoder, for generating VIS-NIR imagery from nighttime MIBT data.}
2.   A FDC and a MFSAtt module are designed. The FDC enhances key features beneficial to the generation task from the frequency-domain perspective, while the MFSAtt adaptively selects critical features in both the spatial and frequency domains that are conducive to improving the quality of the generated images
3.  A set of task-specific loss functions and evaluation metrics, including TIM-MSE loss, GC loss, and TD loss, is proposed to enhance the consistency between the generated results and the ground truth, while comprehensively evaluating the quality of the generated images
4.   An ultra-large-scale dataset is constructed for VIS-NIR image generation. Based on observations from the Himawari-8/9 geostationary meteorological satellites, a dataset containing approximately 70,000 sample groups and nearly 1.21 million images is established, providing a reliable data foundation for related research. 
5.   Comparative experimental results against existing generative AI methods show that RF-VAE achieves superior performance in both image generation quality and inference efficiency, indicating its promising potential for practical applications.

# Dataset VIS-NIR-70K

```python
The VIS-NIR-70K is publicly accessible at https://pan.baidu.com/s/1SVtfqeOLOvTr3gOCDnUOww
```

# Train

```python
python train.py
```

# Test

```python
python test.py
```
