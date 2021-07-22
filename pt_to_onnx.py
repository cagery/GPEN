#!/usr/bin/python3

import torch
import os
import cv2
import glob

import __init_paths
from face_model.face_gan import FaceGAN
from retinaface.retinaface_detection import RetinaFaceDetection

from face_enhancement import FaceEnhancement

def parseToOnnx():

    ## There are two submodels in this model: RetinaFace(facedetector) and FaceGan (facegan)
    ## Both need to be converted to ONNX files (and then used later with MIGraphX, and pre/post optimizations, to get inference example)
    model = {'name':'GPEN-512', 'size':512}
    
    indir = 'examples/imgs'
    outdir = 'examples/outs'
    os.makedirs(outdir, exist_ok=True)

    faceenhancer = FaceEnhancement(size=model['size'], model=model['name'], channel_multiplier=2)

    files = sorted(glob.glob(os.path.join(indir, '*.*g')))
    print(files[0])
    file = files[0]

    im = cv2.imread(file, cv2.IMREAD_COLOR) # BGR
    im = cv2.resize(im, (0,0), fx=2, fy=2)


    facebs, landms = faceenhancer.facedetector.detect(im)
       

    # net = UNet(n_channels=3, n_classes=1)
    # net.load_state_dict(
    #     torch.load('unet_carvana_scale1_epoch5.pth',
    #                map_location=torch.device('cpu')))

    # print(net.eval())
    
    # batch_size, channels, height, width = 1, 3, 256, 256
    # inputs = torch.randn((batch_size, channels, height, width))

    # outputs = net(inputs)
    # assert outputs.shape[0] == batch_size
    # assert not torch.isnan(outputs).any(), 'Output included NaNs'

    # torch.onnx.export(
    #     net,  # model being run
    #     inputs,  # model input (or a tuple for multiple inputs)
    #     "unet_opset13_256.onnx",  # where to save the model (can be a file or file-like   object)
    #     export_params=True,  # store the trained parameter weights inside the model file
    #     opset_version=13,  # the ONNX version to export the model to
    #     do_constant_folding=True,  # whether to execute constant folding for optimization
    #     input_names=['inputs'],  # the model's input names
    #     output_names=['outputs'],  # the model's output names
    #     dynamic_axes={
    #         'inputs': {
    #             0: 'batch_size'
    #         },  # variable lenght axes
    #         'outputs': {
    #             0: 'batch_size'
    #         }
    #     })

    # print("ONNX model conversion is complete.")
    # return

if __name__=='__main__':
    parseToOnnx()
