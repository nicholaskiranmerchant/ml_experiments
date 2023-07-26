import torch, torch.nn as nn
from torchvision import datasets
from torchvision.transforms import ToTensor

def digit_to_onehot(digit : int):
    x = torch.zeros(10)
    x[digit] = 1
    return x

def image_to_vector(image):
    x = ToTensor()(image)
    return torch.flatten(x)

dataset = datasets.MNIST('./data', 
                            download=True, 
                            transform=image_to_vector, 
                            target_transform=digit_to_onehot) 