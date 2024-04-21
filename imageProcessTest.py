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

from detectron2.engine import DefaultTrainer
from detectron2.engine import default_setup
from detectron2.config import get_cfg

from flask import Flask, request
from pyngrok import ngrok, conf
from flask_cors import CORS

import threading

import base64


def setup_cfg(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.MODEL.DEVICE = 'cpu'
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


app = Flask(__name__)
port = "5000"
CORS(app)  # Abilita il CORS per tutte le route

conf.get_default().auth_token = '2eJJC5jmQxIRJtz6L9aMAsyl5hm_7BNTj6dghZRSD7qnpVdrK'

# Open a ngrok tunnel to the HTTP server
public_url = ngrok.connect(port).public_url
print(f" * ngrok tunnel {public_url} -> http://127.0.0.1:{port}")

# Update any base URLs to use the public ngrok URL
app.config["BASE_URL"] = public_url

def encode_image_to_utf8(image_path):
    with open(image_path, "rb") as image_file:
        # Leggi il contenuto dell'immagine binaria
        image_binary = image_file.read()

        # Codifica il contenuto dell'immagine in Base64
        image_base64 = base64.b64encode(image_binary)

        # Converti il risultato in una stringa UTF-8
        image_utf8 = image_base64.decode('utf-8')
        
        return image_utf8

# Define Flask routes
@app.route("/imageProcess", methods=['POST'])
def index():
    data = request.json
    im64 = data.get('image')

    # Decodifica l'immagine base64
    image_bytes = base64.b64decode(im64.split(',')[1])

    path = "testInput.jpg"

    # Scrivi i byte decodificati in un file immagine sul disco
    with open(path, "wb") as image_file:
      image_file.write(image_bytes)

    parser = default_argument_parser()
    args = parser.parse_args("--config-file sign_model_mom/config.yaml".split())
    cfg = setup_cfg(args)

    predictor = DefaultPredictor(cfg)

    files = glob.glob("testInput.jpg")
    sample_size = 1
    for file,_ in zip(files,range(sample_size)):
        im = cv2.imread(file)
        MetadataCatalog.get("mom_dataset_train").thing_classes = ["handwritten"]

        outputs = predictor(im)


        v = Visualizer(im[:, :, ::-1], metadata=MetadataCatalog.get("mom_dataset_train"), scale=1)
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))


        # Codifica il contenuto dell'immagine in base64
        #base64_image = base64.b64encode(v.get_image()).decode('utf-8')

        image_path = "testProcessedCleaned.jpg"
        cv2.imwrite(image_path, v.get_image())
        base64_image = encode_image_to_utf8(image_path)

        result = {"imageProcessed": base64_image}

        return result

# Start the Flask server in a new thread
threading.Thread(target=app.run, kwargs={"use_reloader": False}).start()