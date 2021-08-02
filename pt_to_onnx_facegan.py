#!/usr/bin/python3

import torch
import os
import cv2
import glob
import __init_paths

from face_model.face_gan import FaceGAN

def parseToOnnx():

    ## There are two submodels in this model: RetinaFace(facedetector) and FaceGan (facegan)
    ## Both need to be converted to ONNX files (and then used later with MIGraphX, and pre/post optimizations, to get inference example)
    model = {'name':'GPEN-512', 'size':512}
    
    indir = 'examples/imgs'
    outdir = 'examples/outs'
    os.makedirs(outdir, exist_ok=True)
    base_dir = './'
    model_name = 'GPEN-512'
    size = 512
    channel_multiplier = 2

    facegan = FaceGAN(base_dir, size, model_name, channel_multiplier)
    
    object_methods = [method_name for method_name in dir(facegan)
                  if callable(getattr(facegan, method_name))]
    print(object_methods)

    facegan_model = facegan.model
    
    # Set eval()
    facegan_model.eval()

    # Input definition
    files = sorted(glob.glob(os.path.join(indir, '*.*g')))
    print(files[0])
    file = files[0]
    im = cv2.imread(file, cv2.IMREAD_COLOR) # BGR
    #img = cv2.resize(im, (size, size))
    #img = facegan.img2tensor(img)

    img = cv2.resize(im, (size, size))
    img_t = facegan.img2tensor(img)
    print(f"Input Shape:\t{img_t.shape}")
    print(f"Input Type:\t{type(img_t)}")
    with torch.no_grad():
        out, __ = facegan.model(img_t)
    out = facegan.tensor2img(out)
    print(f"Output Shape:\t{out.shape}")
    print(f"Output Shape:\t{type(out)}")

    # Convert
    torch.onnx.export(
        facegan_model,  # model being run
        img_t,  # model input (or a tuple for multiple inputs)
        "./FaceGAN.onnx",  # where to save the model (can be a file or file-like   object)
        export_params=True,  # store the trained parameter weights inside the model file
        verbose=False,
        opset_version=11,  # the ONNX version to export the model to
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names=['input'],  # the model's input names
        output_names=['output']  # the model's output names
    )

if __name__=='__main__':
    parseToOnnx()
