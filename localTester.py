import glob 
import time
import os 

from matplotlib.pyplot import figure
from matplotlib import pyplot as plt
import cv2

from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

from detectron2.engine import default_argument_parser
from detectron2.engine import DefaultPredictor
import config

parser = default_argument_parser()
args = parser.parse_args("--config-file sign_model_mom/config.yaml".split())
cfg = config.setup_cfg(args)

predictor = DefaultPredictor(cfg)

files = glob.glob("test_images/*.jpg")
sample_size = 5
for file,_ in zip(files,range(sample_size)):
    im = cv2.imread(file)
    MetadataCatalog.get("signature_dataset_train").thing_classes = ["handwritten","handwritten"]
    start_time = time.time()
    outputs = predictor(im)
    print(time.time()- start_time)
    
    v = Visualizer(im[:, :, ::-1], metadata=MetadataCatalog.get("signature_dataset_train"), scale=1)
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    print(file)
    figure(num=None, figsize=(15, 15), dpi=100, facecolor='w', edgecolor='k')
    plt.imshow(v.get_image()[:, :, ::-1])
    plt.show()