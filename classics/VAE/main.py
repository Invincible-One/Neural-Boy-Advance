import os
import sys

import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image



class VAE(nn.Module):
    def __init__(self, img_size, h_dim, z_dim):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(img_size, h_dim)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(h_dim, z_dim)

        self.fc4 = nn.Linear(z_dim, h_dim)
        self.relu2 = nn.ReLU()
        self.fc5 = nn.Linear(h_dim, img_size)
        self.sigmoid = nn.Sigmoid()

    def _encode(self, x):
        h = self.relu1(self.fc1(x))
        return self.fc2(h), self.fc3(h)

    def _reparameterize(self, mu, log_var):
        std = torch.exp(log_var / 2)
        eps = torch.randn_like(std)
        return mu + eps * std

    def _decode(self, z):
        x = self.sigmoid(self.fc5(self.relu2(self.fc4(z))))
        return x

    def forward(self, x):
        mu, log_var = self._encode(x)
        z = self._reparameterize(mu, log_var)
        x = self._decode(z)
        return x, mu, log_var



if __name__ == "__main__":
    img_size = 784
    h_dim = 400
    z_dim = 20
    epochs = 15
    batch_size = 128
    lr = 1e-3

    device = torch.device('cuda')
    data_root = "/scratch/ym2380/data/mnist/"
    training_set = torchvision.datasets.MNIST(root=data_root, train=True, transform=transforms.ToTensor(), download=True)
    training_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True)
    
    working_dir = "./saved/"
    if not os.path.exists(working_dir):
        os.makedirs(working_dir)

    network = VAE(img_size, h_dim, z_dim)
    network.to(device)
    loss_f = nn.BCELoss()
    loss_f.to(device)

    optimizer = optim.Adam(network.parameters(), lr=lr)

    count = 0
    for _ in range(epochs):
        count += 1
        for i, (X, _) in enumerate(training_loader):
            X = X.to(device).view(-1, img_size)
            X_hat, mu, log_var = network(X)
            loss_v = loss_f(X_hat, X)
            kl_div = - 0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            loss = loss_v + kl_div 
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch[{count}/{epochs}], Recons_Loss: {loss_v.item():.4f}, KL Div: {kl_div.item():.4f}")

        with torch.no_grad():
            z = torch.randn(batch_size, z_dim).to(device)
            X_hat = network._decode(z).view(-1, 1, 28, 28)
            save_image(X_hat, os.path.join(working_dir, f"sampled_{count}.png"))
