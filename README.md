# Image Caption

This is an implementation of *Bridging by Word: Image-Grounded Vocabulary Construction for Visual Captioning* based on pytorch 1.0.

## Setup
0. Install python3.6 and pytorch 1.0.
1. Download the image data of MS COCO to *Data/raw/img*.

2. Data process, you need to create a folder in *Data/train* with the name of *coco_v?* to store the processed data.
```python3
cd Code
python BuildImgOrderClusterVocab.py
```

3. Model training. 
```python3
cd Code
python Trainer.py 
```
