

from datetime import datetime
import pickle
import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import logging
from models.transformer_film_cond import TransformerDDPME
from PIL import Image


class Diffusion:

    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, batch_size=64, vocab_size=2048, time_steps=16):

        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.time_steps = time_steps

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.beta = self.prepare_noise_schedule().to(self.device)
        self.alpha = 1.0 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_latents(self, x, t):

        t_squeeze = t.squeeze(-1)
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t_squeeze])[:, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t_squeeze])[:, None, None]
        eps = torch.randn_like(x)

        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * eps, eps

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n, 1))

    def sample(self, model, n, labels, cfg_scale=3):
        logging.info(f"sampling {n} new latents...")
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, self.time_steps, self.vocab_size)).to(self.device)
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
            # for i in reversed(range(1, self.noise_steps)):

                t = (torch.ones(n)*i).long().to(self.device)
                t_expand = t[:, None]
                predicted_noise = model(x, t_expand, torch.tensor([labels]))
                if cfg_scale > 0:
                    uncond_predicted_noise = model(x, t_expand, None)
                    predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)

                alpha = self.alpha[t][:, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None]
                beta = self.beta[t][:, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)

                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise

            model.train()
            x = (x.clamp(-1, 1) + 1) / 2
            x = (x * 255).type(torch.uint8)
            return x

def train():


    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        for i in range(num_gpus):
            gpu_name = torch.cuda.get_device_name(i)
            print(f"GPU {i}: {gpu_name}")

    lr = 5e-4
    batch_size = 256

    epochs = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    categories = {"numbers": 10}
    model = TransformerDDPME(categories).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=4000//(batch_size//64), gamma=0.98)
    mse = nn.MSELoss()


    diffusion = Diffusion(noise_steps=model.num_timesteps, batch_size=batch_size, vocab_size=model.vocab_size, time_steps=model.seq_len)

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    # Load the MNIST dataset with the defined transform
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)

    # Create a dataloader with a batch size of 64
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    train_losses = []

    os.makedirs("./generated", exist_ok=True)

    for epoch in range(epochs):


        pbar = tqdm(train_loader)
        # pbar = train_loader

        train_count = 0
        train_loss_sum = 0

        for step, (images, labels) in enumerate(pbar):

            images = images.squeeze(1)
            images = images.to(device)
            labels = labels.to(device)

            if np.random.random() < 0.1:
                labels = None

            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            x_t, noise = diffusion.noise_latents(images, t)

            predicted_noise = model(x_t, t, labels)

            loss = mse(noise, predicted_noise)
            train_loss_sum += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_count += 1

        mean_train_loss = train_loss_sum / train_count
        train_losses.append(mean_train_loss)
        current_lr = optimizer.param_groups[0]['lr']
        logging.info(f"Learning rate at epoch  {epoch}:{current_lr}")
        logging.info(f"Epoch {epoch} mean training loss: {mean_train_loss}")

        sampled_latents = diffusion.sample(model, 1, 5, cfg_scale=3)
        sampled_latents = sampled_latents.numpy()
        sampled_latents = sampled_latents.squeeze(0)
        im = Image.fromarray(sampled_latents)
        im.save(f"./generated/{epoch}.jpg")


        checkpoint = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict(),
                      "epoch": (epoch)}
        torch.save(checkpoint, "./last_checkpoint.pth.tar")

        epochs = range(len(train_losses))
        plt.plot(epochs, train_losses, 'r', label='Training Loss')

        # Add labels and a legend
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Validation and Training Losses')
        plt.legend()
        loss_plot_abs_path = f"./loss_plot_{epoch}.png"
        plt.savefig(loss_plot_abs_path)
        plt.clf()



if __name__ == "__main__":
    train()
