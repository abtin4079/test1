import sys
sys.path.append('/')

import cv2
import numpy as np
from torchvision import models
from efficientnet_pytorch import EfficientNet
from grad_cam import GradCam, GuidedBackpropReLUModel, show_cams, show_gbs, preprocess_image
import argparse

def VGG(parser):
    model = models.vgg19(pretrained=True)
    grad_cam = GradCam(model=model, blob_name = 'features', target_layer_names=['36'], use_cuda=False)
    img = cv2.imread(parser.image_path, 1)
    img = np.float32(cv2.resize(img, (224, 224))) / 255
    inputs = preprocess_image(img)
    # If None, returns the map for the highest scoring category.
    # Otherwise, targets the requested index.
    target_index = None
    mask_dic = grad_cam(inputs, target_index)
    show_cams(img, mask_dic, parser.image_number)
    gb_model = GuidedBackpropReLUModel(model=model, activation_layer_name = 'ReLU', use_cuda=False)
    #show_gbs(inputs, gb_model, target_index, mask_dic)



def effienctnet(parser):
    model = EfficientNet.from_pretrained('efficientnet-b0')
    grad_cam = GradCam(model=model, blob_name = '_blocks', target_layer_names=['15'], use_cuda=False)
    img = cv2.imread(parser.image_path, 1)
    img = np.float32(cv2.resize(img, (224, 224))) / 255
    inputs = preprocess_image(img)
    # If None, returns the map for the highest scoring category.
    # Otherwise, targets the requested index.
    target_index = None
    mask_dic = grad_cam(inputs, target_index)
    show_cams(img, mask_dic, parser.image_number)
    gb_model = GuidedBackpropReLUModel(model=model, activation_layer_name = 'GuidedBackpropSwish', use_cuda=False)
    #show_gbs(inputs, gb_model, target_index, mask_dic)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, default=None)
    parser.add_argument('--image_number', type=str, default=None)
    #parser.add_argument('--pretrained_weight_path', type=str, default=None)
    parser = parser.parse_args()
    #VGG(parser)
    effienctnet(parser)


    # comment this 