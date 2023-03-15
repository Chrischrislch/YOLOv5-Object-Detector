import argparse
import io
import os
from PIL import Image
import datetime

import torch
from flask import Flask, render_template, request, redirect
app = Flask(__name__)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)  # force_reload = recache latest code
model.eval()

@app.route('/')
def index():
    return render_template("index.html")

@app.route("/", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)
        file = request.files["file"]
        if not file:
            return
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes))
        results = model([img])
        results.render()  
        now_time = datetime.datetime.now().strftime("%m-%d-%Y")
        img_savename = f"static/{now_time}.png"
        Image.fromarray(results.ims[0]).save(img_savename)
        return redirect(img_savename)
    
    return render_template("index.html")
