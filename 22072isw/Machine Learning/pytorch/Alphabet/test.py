import os
import torch
import torchvision.models as models
import cv2
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from classes import NeuralNetwork

def image_transition(dir) :
    try :
        img = cv2.imread(dir, 0)
        if img.shape != (28, 28) :
            img = cv2.resize(img, dsize=(28, 28), interpolation=cv2.INTER_CUBIC)
        img_array = np.array(img).astype('float32')
    except :
        img_array = np.array([[0] * 28 for _ in range(28)])
    return img_array

device = 'cuda'

model = NeuralNetwork()
saved_path = 'model_weights.pth'
if os.path.exists(saved_path) :
    model.load_state_dict(torch.load(saved_path))
    model.eval()
model.to(device)
    
for f in os.listdir('./test_images') :
    fpath = os.path.join('./test_images', f)
    image = image_transition(fpath)
    image_tensor = torch.tensor(image).to(device)
    flatten = nn.Flatten(0, -1).to(device)
    image_tensor = flatten(image_tensor)
    predict = model(image_tensor)
    print("File Name : {0}  |  Result : {1}".format(f, chr(predict.argmax(0) + 97)))