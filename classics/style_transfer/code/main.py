import os
import sys
import argparse
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision
import torch.nn as nn
from torchvision import transforms
from torchvision import models



def load_img(img_path, transform=None, max_size=None, shape=None):
    img = Image.open(img_path)

    if max_size:
        scale = max_size / max(img.size)
        size = np.array(img.size) * scale
        img = img.resize(size.astype(int), Image.ANTIALIAS)
    if shape:
        img = img.resize(shape, Image.LANCZOS)
    if transform:
        img = transform(img).unsqueeze(0)

    return img.to(device)



class VGG19(nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()
        self.select = ['0', '5', '10', '19', '28']
        self.vgg = models.vgg19(pretrained=True).features

    def forward(self, x):
        features = []
        for name, layers in self.vgg._modules.items():
            x = layers(x)
            if name in self.select:
                features.append(x)
        return features



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--content", type=str, default="content001.png")
    parser.add_argument("--style", type=str, default="style001.png")
    parser.add_argument("--max_size", type=int, default=400)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--style_weight", type=float, default=100)
    parser.add_argument("--working_dir", type=str, default="content001_style001/")
    params = parser.parse_args()
    content = params.content
    style = params.style
    max_size = params.max_size
    lr = params.lr
    epochs = params.epochs
    style_weight = params.style_weight
    working_dir = params.working_dir

    print(f'''
    *******************************INFO*******************************
    It has been 6 years. Can you believe it?
    Content: {content}
    Style: {style}
    max_size: {max_size}
    lr: {lr}
    epochs: {epochs}
    style_weight: {style_weight}
    working_dir: {working_dir}
    ******************************************************************
    ''')

    img_dir = "../images/"
    working_dir = os.path.join("../saved/", working_dir)
    if not os.path.exists(working_dir):
        os.makedirs(working_dir)

    content = os.path.join(img_dir, content)
    style = os.path.join(img_dir, style)
    
    device = torch.device('cuda')

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                                                std=(0.229, 0.224, 0.225))])

    content_img = load_img(content, transform=transform, max_size=max_size)
    style_img = load_img(style, transform=transform, shape=(content_img.shape[2], content_img.shape[3]))
    target_img = content_img.clone().requires_grad_(True)

    network = VGG19()
    network.to(device)
    network.eval()
    optimizer = torch.optim.Adam([target_img], lr=lr, betas=[0.5, 0.999])
    
    count = 0
    for _ in range(epochs):
        count += 1

        content_features = network(content_img)
        style_features = network(style_img)
        target_features = network(target_img)

        style_loss = 0
        content_loss = 0
        for f1, f2, f3 in zip(target_features, content_features, style_features):
            content_loss += torch.mean((f1 - f2)**2)
            
            _, c, h, w = f1.size()
            f1 = f1.view(c, h * w)
            f3 = f3.view(c, h * w)

            f1 = torch.mm(f1, f1.t())
            f3 = torch.mm(f3, f3.t())

            style_loss += torch.mean((f1 - f3)**2) / (c * h * w)

        loss = content_loss + style_weight * style_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if count % 10  == 0:
            print ('Epoch [{}/{}], Content Loss: {:.4f}, Style Loss: {:.4f}'
                   .format(count, epochs, content_loss.item(), style_loss.item()))

        if count % 100 == 0:
            denorm = transforms.Normalize((-2.12, -2.04, -1.80), (4.37, 4.46, 4.44))
            img = target_img.clone().squeeze()
            img = denorm(img).clamp_(0, 1)
            target = os.path.join(working_dir, f"output_{count}.png")
            torchvision.utils.save_image(img, target)
