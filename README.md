# Image Caption

This is an implementation of *Bridging by Word: Image-Grounded Vocabulary Construction for Visual Captioning* based on pytorch 1.0.

## Run
1.Download the image data of MS COCO to *Data/raw/img*.

2.Data process
```python3
python BuildImgOrderClusterVocab.py
```

3.Model training
```python3
python Trainer.py 
```
