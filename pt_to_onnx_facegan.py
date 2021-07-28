#!/usr/bin/python3

import torch
import os
import cv2
import glob
import numpy as np
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


    print(facegan)
    object_methods = [method_name for method_name in dir(facedetector)
                  if callable(getattr(facedetector, method_name))]
    print(object_methods)
    
    raise

    # Set eval()
    facedetector.net.eval()

    # Input definition
    files = sorted(glob.glob(os.path.join(indir, '*.*g')))
    print(files[0])
    file = files[0]
    im = cv2.imread(file, cv2.IMREAD_COLOR) # BGR
    im = cv2.resize(im, (0,0), fx=2, fy=2)

    # Input dimensions:
    img = np.float32(im)
    im_height, im_width = img.shape[:2]
    scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
    img -= (104, 117, 123)
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).unsqueeze(0)
    img = img.cuda()
    scale = scale.cuda()

    # Run forward pass and get input dimensions.
    # loc,conf,landsm are all torch.Tensor outputs.
    loc, conf, landms = facedetector.net(img)
    print(img.shape)    # torch.Size([1, 3, 1136, 2656])
    print(loc.shape)    # torch.Size([1, 123836, 4])
    print(conf.shape)   # torch.Size([1, 123836, 2])
    print(landms.shape) # torch.Size([1, 123836, 10])

    # Convert
    torch.onnx.export(
        facedetector.net,  # model being run
        img,  # model input (or a tuple for multiple inputs)
        "RetinaFace.onnx",  # where to save the model (can be a file or file-like   object)
        export_params=True,  # store the trained parameter weights inside the model file
        opset_version=13,  # the ONNX version to export the model to
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names=['inputs'],  # the model's input names
        output_names=['loc','conf','landms'],  # the model's output names
        dynamic_axes={
            'inputs': {
                0: 'batch_size'
            },  'loc': {
                    0: 'batch_size'
            },  'conf': {
                    0: 'batch_size'
            },  'landms': {
                    0: 'batch_size'
            }
        })

if __name__=='__main__':
    parseToOnnx()
