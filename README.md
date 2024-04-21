# AIHandwrittenDetection

# "Detecting Handwritten text in Documents"
> Detect signature and handwritten text in images using Detectron2 library. We will train a custom object detection model and run the inferences on the images
- toc: true
- branch: master
- badges: true
- comments: true
- categories: [detectron2, FasterRCNN, vision]


### Training Dataset
For the training data, we have the labeled dataset available with us so we don't have to do the tedious work of labeling each handwritten text.

We will use the dataset available here
https://github.com/CatalystCode/Handwriting/tree/master/Data/labelledcontracttrainingdata/trainingjpg_output_99/

The dataset is part of the Microsoft blog available here
https://devblogs.microsoft.com/cse/2018/05/07/handwriting-detection-and-recognition-in-scanned-documents-using-azure-ml-package-computer-vision-azure-cognitive-services-ocr/
