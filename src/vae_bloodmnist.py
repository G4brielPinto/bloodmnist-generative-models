"""
Variational Autoencoder (VAE) Implementation for BloodMNIST Dataset

This module provides a comprehensive implementation of a Convolutional Variational
Autoencoder (ConvVAE) for the BloodMNIST dataset. It includes modular architecture
components, training utilities, evaluation metrics, and visualization tools.

The implementation follows the project requirements for Advanced Machine Learning TP2,
including proper FID evaluation methodology and result documentation.

Author: Gabriel Pinto & João Antunes
"""

import os
import time
import json
import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.tensorboard import SummaryWriter

from torchvision import transforms, utils
from torchvision.utils import save_image, make_grid
from torchmetrics.image.fid import FrechetInceptionDistance

from medmnist import BloodMNIST


class ConvEncoder(nn.Module):
    """
    Convolutional Encoder for the VAE.

    Transforms input images into latent space parameters (mean and log variance).
    The encoder consists of a series of convolutional layers with batch normalization
    and LeakyReLU activations, followed by global average pooling and projection
    to the latent space parameters.

    Args:
        in_channels (int): Number of input image channels (default: 3 for RGB)
        hidden_dims (list): List of hidden dimensions for convolutional layers
        latent_dim (int): Dimension of the latent space
        img_size (int): Input image size (assumed square)
    """
    def __init__(self, in_channels=3, hidden_dims=[32, 64, 128, 256], latent_dim=128, img_size=28):
        super().__init__()

        self.latent_dim = latent_dim
        self.img_size = img_size
        self.hidden_dims = hidden_dims
        self.in_channels = in_channels

        modules = []

        # Build encoder layers
        current_channels = in_channels
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(current_channels, h_dim, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU(0.2)
                )
            )
            current_channels = h_dim

        self.encoder = nn.Sequential(*modules)

        # Use Global Average Pooling instead of flatten for better feature extraction
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Projection to latent space parameters (mean and log variance)
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)

    def forward(self, x):
        """
        Forward pass through the encoder.

        Args:
            x (torch.Tensor): Input tensor of shape [B, C, H, W]

        Returns:
            tuple: (mu, logvar) tensors representing the latent space parameters
                  mu: mean of the latent Gaussian distribution
                  logvar: log variance of the latent Gaussian distribution
        """
        x = self.encoder(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)  # Flatten to [B, hidden_dims[-1]]

        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)

        return mu, logvar


class ConvDecoder(nn.Module):
    """
    Convolutional Decoder for the VAE.

    Transforms latent vectors back into images. The decoder consists of an initial
    linear projection, followed by a series of transposed convolutional layers with
    batch normalization and LeakyReLU activations, and a final convolution with
    Tanh activation to produce the output image.

    Args:
        latent_dim (int): Dimension of the latent space
        hidden_dims (list): List of hidden dimensions for convolutional layers (in reverse)
        output_channels (int): Number of output image channels (default: 3 for RGB)
        img_size (int): Output image size (assumed square)
    """
    def __init__(self, latent_dim=128, hidden_dims=[256, 128, 64, 32], output_channels=3, img_size=28):
        super().__init__()

        self.latent_dim = latent_dim
        self.img_size = img_size
        self.hidden_dims = hidden_dims

        # Calculate initial size after deconvolutions
        num_layers = len(hidden_dims)
        self.init_size = max(1, img_size // (2 ** num_layers))
        self.init_channels = hidden_dims[0]

        # Initial linear projection from latent space to initial feature map
        self.fc_decoder = nn.Linear(
            latent_dim,
            self.init_channels * self.init_size * self.init_size
        )

        modules = []

        # Build decoder layers with transposed convolutions
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        hidden_dims[i],
                        hidden_dims[i + 1],
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        output_padding=1
                    ),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU(0.2)
                )
            )

        # Final upsampling layer to ensure correct output size
        modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(
                    hidden_dims[-1],
                    hidden_dims[-1],
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1
                ),
                nn.BatchNorm2d(hidden_dims[-1]),
                nn.LeakyReLU(0.2)
            )
        )

        # Final convolution to get the right number of channels with Tanh activation
        modules.append(
            nn.Sequential(
                nn.Conv2d(
                    hidden_dims[-1],
                    output_channels,
                    kernel_size=3,
                    padding=1
                ),
                nn.Tanh()  # Output in range [-1, 1] to match normalized input
            )
        )

        self.decoder = nn.Sequential(*modules)

        # Add a final resize layer to ensure exact output size
        self.final_resize = nn.Upsample(size=(img_size, img_size), mode='bilinear', align_corners=False)

    def forward(self, z):
        """
        Forward pass through the decoder.

        Args:
            z (torch.Tensor): Latent vector of shape [B, latent_dim]

        Returns:
            torch.Tensor: Reconstructed image tensor of shape [B, C, H, W]
        """
        x = self.fc_decoder(z)
        x = x.view(-1, self.init_channels, self.init_size, self.init_size)
        x = self.decoder(x)

        # Ensure output size is exactly img_size x img_size
        x = self.final_resize(x)

        return x


class ConvVAE(nn.Module):
    """
    Convolutional Variational Autoencoder (VAE) for image generation.

    Combines the encoder and decoder into a complete VAE model. The VAE learns to
    encode images into a probabilistic latent space and decode samples from this
    space back into images. The model is trained using a combination of reconstruction
    loss and KL divergence, weighted by a beta parameter.

    Args:
        in_channels (int): Number of input image channels
        latent_dim (int): Dimension of the latent space
        hidden_dims (list): List of hidden dimensions for convolutional layers
        img_size (int): Input/output image size (assumed square)
        beta (float): Weight for the KL divergence term in the loss function
                     (higher values enforce more regularized latent space)
    """
    def __init__(
        self,
        in_channels=3,
        latent_dim=128,
        hidden_dims=[32, 64, 128, 256],
        img_size=28,
        beta=1.0
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.beta = beta
        self.img_size = img_size

        # Create encoder and decoder components
        self.encoder = ConvEncoder(
            in_channels=in_channels,
            hidden_dims=hidden_dims,
            latent_dim=latent_dim,
            img_size=img_size
        )

        self.decoder = ConvDecoder(
            latent_dim=latent_dim,
            hidden_dims=list(reversed(hidden_dims)),
            output_channels=in_channels,
            img_size=img_size
        )

    def encode(self, x):
        """
        Encode input to latent parameters.

        Args:
            x (torch.Tensor): Input tensor of shape [B, C, H, W]

        Returns:
            tuple: (mu, logvar) tensors representing the latent space parameters
        """
        return self.encoder(x)

    def decode(self, z):
        """
        Decode latent vector to image.

        Args:
            z (torch.Tensor): Latent vector of shape [B, latent_dim]

        Returns:
            torch.Tensor: Reconstructed image tensor of shape [B, C, H, W]
        """
        return self.decoder(z)

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to enable backpropagation through the sampling process.

        This technique allows gradients to flow through the stochastic sampling operation
        by separating the deterministic and stochastic components.

        Args:
            mu (torch.Tensor): Mean of the latent Gaussian distribution
            logvar (torch.Tensor): Log variance of the latent Gaussian distribution

        Returns:
            torch.Tensor: Sampled latent vector
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def forward(self, x):
        """
        Forward pass through the VAE.

        Args:
            x (torch.Tensor): Input tensor of shape [B, C, H, W]

        Returns:
            tuple: (reconstruction, input, mu, logvar) tensors
                  reconstruction: Reconstructed image
                  input: Original input image
                  mu: Mean of the latent Gaussian distribution
                  logvar: Log variance of the latent Gaussian distribution
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)

        return reconstruction, x, mu, logvar

    def sample(self, num_samples, device):
        """
        Generate samples from the latent space.

        Args:
            num_samples (int): Number of samples to generate
            device (torch.device): Device to generate samples on

        Returns:
            torch.Tensor: Generated samples of shape [num_samples, C, H, W]
        """
        z = torch.randn(num_samples, self.latent_dim).to(device)
        samples = self.decode(z)
        return samples

    def loss_function(self, recon_x, x, mu, logvar):
        """
        VAE loss function combining reconstruction loss and KL divergence.

        The loss consists of two components:
        1. Reconstruction loss: Measures how well the model reconstructs the input
           (combination of MSE and L1 loss)
        2. KL divergence: Regularizes the latent space to follow a standard normal distribution

        Args:
            recon_x (torch.Tensor): Reconstructed images
            x (torch.Tensor): Original images
            mu (torch.Tensor): Mean of the latent Gaussian distribution
            logvar (torch.Tensor): Log variance of the latent Gaussian distribution

        Returns:
            tuple: (total_loss, reconstruction_loss, kl_divergence)
        """
        # Ensure recon_x has the same size as x
        if recon_x.shape != x.shape:
            recon_x = F.interpolate(recon_x, size=(self.img_size, self.img_size), mode='bilinear', align_corners=False)

        # Reconstruction loss (weighted combination of MSE and L1)
        mse_loss = F.mse_loss(recon_x, x, reduction='sum')
        l1_loss = F.l1_loss(recon_x, x, reduction='sum')
        recon_loss = 0.7 * mse_loss + 0.3 * l1_loss

        # KL divergence: -0.5 * sum(1 + log(σ^2) - μ^2 - σ^2)
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        # Total loss with beta weighting for KL divergence
        total_loss = recon_loss + self.beta * kl_div

        return total_loss, recon_loss, kl_div


class VAETrainer:
    """
    Trainer class for the VAE model.

    Handles training, evaluation, visualization, and saving of results.

    Args:
        model (nn.Module): VAE model to train
        train_loader (DataLoader): DataLoader for training data
        val_loader (DataLoader): DataLoader for validation data
        optimizer (torch.optim.Optimizer): Optimizer for training
        scheduler (torch.optim.lr_scheduler._LRScheduler, optional): Learning rate scheduler
        device (torch.device): Device to train on
        config (dict): Configuration dictionary with hyperparameters
        output_dir (str): Directory to save results
    """
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler=None,
        device=None,
        config=None,
        output_dir="./results"
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config if config is not None else {}

        # Setup directories for saving results
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.model_name = self.config.get("model_name", "ConvVAE")
        self.output_dir = os.path.join(output_dir, f"{self.model_name}_{self.timestamp}")

        self.model_dir = os.path.join(self.output_dir, "models")
        self.images_dir = os.path.join(self.output_dir, "images")
        self.plots_dir = os.path.join(self.output_dir, "plots")
        self.logs_dir = os.path.join(self.output_dir, "logs")

        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)

        # Setup TensorBoard for logging
        self.writer = SummaryWriter(self.logs_dir)

        # Training metrics
        self.train_losses = []
        self.recon_losses = []
        self.kl_losses = []
        self.fid_scores = []
        self.best_fid = float('inf')
        self.best_epoch = 0

        # Save configuration
        self.save_config()

    def save_config(self):
        """
        Save configuration to a JSON file.

        This allows for reproducibility and tracking of hyperparameters.
        """
        config_path = os.path.join(self.output_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(self.config, f, indent=4)

    def train_epoch(self, epoch):
        """
        Train for one epoch.

        Args:
            epoch (int): Current epoch number

        Returns:
            tuple: (average_loss, average_recon_loss, average_kl_loss)
        """
        self.model.train()
        total_loss = 0
        recon_loss = 0
        kl_loss = 0

        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}")

        for batch_idx, (data, _) in enumerate(progress_bar):
            data = data.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            recon_batch, original_batch, mu, logvar = self.model(data)
            loss, r_loss, k_loss = self.model.loss_function(recon_batch, original_batch, mu, logvar)

            # Backward pass and optimization
            loss.backward()
            self.optimizer.step()

            # Track losses
            total_loss += loss.item()
            recon_loss += r_loss.item()
            kl_loss += k_loss.item()

            progress_bar.set_postfix(loss=loss.item()/len(data))

        # Calculate average losses
        avg_loss = total_loss / len(self.train_loader.dataset)
        avg_recon = recon_loss / len(self.train_loader.dataset)
        avg_kl = kl_loss / len(self.train_loader.dataset)

        # Log to TensorBoard
        self.writer.add_scalar('Loss/train', avg_loss, epoch)
        self.writer.add_scalar('Reconstruction_Loss/train', avg_recon, epoch)
        self.writer.add_scalar('KLD_Loss/train', avg_kl, epoch)

        return avg_loss, avg_recon, avg_kl

    def compute_fid(self, num_samples=2000, batch_size=64):
        """
        Compute FID score between real and generated images.

        The Fréchet Inception Distance (FID) measures the similarity between
        two sets of images by comparing the statistics of their feature representations
        extracted from a pre-trained InceptionV3 network.

        Args:
            num_samples (int): Number of samples to use for FID calculation
            batch_size (int): Batch size for processing

        Returns:
            float: FID score (lower is better)
        """
        self.model.eval()
        fid = FrechetInceptionDistance(feature=2048, normalize=True).to(self.device)

        # Process real images
        real_processed = 0
        with torch.no_grad():
            for data, _ in self.val_loader:
                data = data.to(self.device)
                # Scale from [-1, 1] to [0, 255] for FID calculation
                data_scaled = ((data + 1) * 127.5).to(torch.uint8)
                fid.update(data_scaled, real=True)

                real_processed += data.shape[0]
                if real_processed >= num_samples:
                    break

        # Process generated images
        for i in range(0, num_samples, batch_size):
            with torch.no_grad():
                current_batch_size = min(batch_size, num_samples - i)
                samples = self.model.sample(current_batch_size, self.device)
                # Scale from [-1, 1] to [0, 255] for FID calculation
                samples_scaled = ((samples + 1) * 127.5).to(torch.uint8)
                fid.update(samples_scaled, real=False)

        # Calculate FID score
        fid_score = fid.compute().item()
        return fid_score

    def validate(self, epoch):
        """
        Validate the model on the validation set.

        Args:
            epoch (int): Current epoch number

        Returns:
            tuple: (average_loss, average_recon_loss, average_kl_loss)
        """
        self.model.eval()
        total_loss = 0
        recon_loss = 0
        kl_loss = 0

        with torch.no_grad():
            for batch_idx, (data, _) in enumerate(self.val_loader):
                data = data.to(self.device)

                # Forward pass
                recon_batch, original_batch, mu, logvar = self.model(data)
                loss, r_loss, k_loss = self.model.loss_function(recon_batch, original_batch, mu, logvar)

                # Track losses
                total_loss += loss.item()
                recon_loss += r_loss.item()
                kl_loss += k_loss.item()

        # Calculate average losses
        avg_loss = total_loss / len(self.val_loader.dataset)
        avg_recon = recon_loss / len(self.val_loader.dataset)
        avg_kl = kl_loss / len(self.val_loader.dataset)

        # Log to TensorBoard
        self.writer.add_scalar('Loss/val', avg_loss, epoch)
        self.writer.add_scalar('Reconstruction_Loss/val', avg_recon, epoch)
        self.writer.add_scalar('KLD_Loss/val', avg_kl, epoch)

        return avg_loss, avg_recon, avg_kl

    def save_checkpoint(self, epoch, is_best=False):
        """
        Save model checkpoint.

        Args:
            epoch (int): Current epoch number
            is_best (bool): Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'recon_losses': self.recon_losses,
            'kl_losses': self.kl_losses,
            'fid_scores': self.fid_scores,
            'best_fid': self.best_fid,
            'best_epoch': self.best_epoch
        }

        # Save latest checkpoint
        torch.save(checkpoint, os.path.join(self.model_dir, f"checkpoint_latest.pt"))

        # Save epoch checkpoint
        if epoch % 50 == 0:
            torch.save(checkpoint, os.path.join(self.model_dir, f"checkpoint_epoch_{epoch}.pt"))

        # Save best model
        if is_best:
            torch.save(checkpoint, os.path.join(self.model_dir, f"checkpoint_best.pt"))

    def save_samples(self, epoch, num_samples=16, is_final=False):
        """
        Generate and save samples from the model.

        Args:
            epoch (int): Current epoch number
            num_samples (int): Number of samples to generate
            is_final (bool): Whether these are the final samples
        """
        self.model.eval()
        with torch.no_grad():
            # Generate samples
            samples = self.model.sample(num_samples, self.device)

            # Save grid of samples
            if is_final:
                save_dir = os.path.join(self.images_dir, "final_samples")
                os.makedirs(save_dir, exist_ok=True)

                # Save individual samples
                for i, sample in enumerate(samples):
                    save_image(sample, os.path.join(save_dir, f"sample_{i}.png"), normalize=True)

                # Save grid
                grid = make_grid(samples, nrow=4, normalize=True)
                save_image(grid, os.path.join(self.images_dir, f"samples_final.png"))
            else:
                grid = make_grid(samples, nrow=4, normalize=True)
                save_image(grid, os.path.join(self.images_dir, f"samples_epoch_{epoch}.png"))

            # Log to TensorBoard
            self.writer.add_image('Generated_Samples', grid, epoch)

    def save_reconstructions(self, epoch, num_samples=8, is_final=False):
        """
        Generate and save reconstructions from the validation set.

        Args:
            epoch (int): Current epoch number
            num_samples (int): Number of samples to reconstruct
            is_final (bool): Whether these are the final reconstructions
        """
        self.model.eval()
        with torch.no_grad():
            # Get samples from validation set
            data_iter = iter(self.val_loader)
            data, _ = next(data_iter)
            data = data[:num_samples].to(self.device)

            # Reconstruct
            recon, _, _, _ = self.model(data)

            # Create comparison grid
            comparison = torch.cat([data, recon])
            grid = make_grid(comparison, nrow=num_samples, normalize=True)

            # Save grid
            if is_final:
                save_image(grid, os.path.join(self.plots_dir, f"final_comparison.png"))
            else:
                save_image(grid, os.path.join(self.plots_dir, f"comparison_epoch_{epoch}.png"))

            # Log to TensorBoard
            self.writer.add_image('Reconstructions', grid, epoch)

    def plot_losses(self):
        """
        Plot training losses and save the figure.
        """
        plt.figure(figsize=(12, 8))

        epochs = range(1, len(self.train_losses) + 1)

        plt.subplot(3, 1, 1)
        plt.plot(epochs, self.train_losses, 'b-', label='Total Loss')
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        plt.subplot(3, 1, 2)
        plt.plot(epochs, self.recon_losses, 'r-', label='Reconstruction Loss')
        plt.title('Reconstruction Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        plt.subplot(3, 1, 3)
        plt.plot(epochs, self.kl_losses, 'g-', label='KL Divergence')
        plt.title('KL Divergence')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, "training_losses.png"), dpi=300)
        plt.close()

    def plot_fid_scores(self):
        """
        Plot FID scores and save the figure.
        """
        if not self.fid_scores:
            return

        plt.figure(figsize=(10, 6))

        fid_epochs = range(self.config.get("fid_interval", 5),
                          len(self.fid_scores) * self.config.get("fid_interval", 5) + 1,
                          self.config.get("fid_interval", 5))

        plt.plot(fid_epochs, self.fid_scores, 'b-o', label='FID Score')
        plt.title('FID Score Evolution')
        plt.xlabel('Epoch')
        plt.ylabel('FID Score (lower is better)')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, "fid_scores.png"), dpi=300)
        plt.close()

    def save_results(self, training_time):
        """
        Save final results to a JSON file.

        Args:
            training_time (dict): Dictionary with training time information
        """
        results = {
            "model_name": self.model_name,
            "best_fid": self.best_fid,
            "best_epoch": self.best_epoch,
            "final_loss": self.train_losses[-1] if self.train_losses else None,
            "final_recon_loss": self.recon_losses[-1] if self.recon_losses else None,
            "final_kl_loss": self.kl_losses[-1] if self.kl_losses else None,
            "fid_scores": self.fid_scores,
            "training_time": training_time
        }

        results_path = os.path.join(self.output_dir, "results.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=4)

    def compute_final_fid(self, num_runs=5, num_samples=10000, seed=42):
        """
        Compute final FID score over multiple runs.

        This follows the project evaluation methodology:
        - 10,000 real samples
        - 10,000 generated samples
        - 5 independent runs with different seeds

        Args:
            num_runs (int): Number of independent runs
            num_samples (int): Number of samples per run
            seed (int): Base seed for reproducibility

        Returns:
            tuple: (mean_fid, std_fid, all_fids)
        """
        print(f"Computing final FID score over {num_runs} runs with {num_samples} samples each...")

        self.model.eval()
        all_fids = []

        for run in range(num_runs):
            # Set seed for reproducibility
            current_seed = seed + run
            torch.manual_seed(current_seed)
            np.random.seed(current_seed)

            # Initialize FID
            fid = FrechetInceptionDistance(feature=2048, normalize=True).to(self.device)

            # Process real images
            real_processed = 0
            with torch.no_grad():
                for data, _ in self.val_loader:
                    data = data.to(self.device)
                    # Scale from [-1, 1] to [0, 255]
                    data_scaled = ((data + 1) * 127.5).to(torch.uint8)
                    fid.update(data_scaled, real=True)

                    real_processed += data.shape[0]
                    if real_processed >= num_samples:
                        break

            # Process generated images
            batch_size = 64
            for i in range(0, num_samples, batch_size):
                with torch.no_grad():
                    current_batch_size = min(batch_size, num_samples - i)
                    samples = self.model.sample(current_batch_size, self.device)
                    # Scale from [-1, 1] to [0, 255]
                    samples_scaled = ((samples + 1) * 127.5).to(torch.uint8)
                    fid.update(samples_scaled, real=False)

            # Calculate FID score
            fid_score = fid.compute().item()
            all_fids.append(fid_score)
            print(f"Run {run+1}/{num_runs}: FID = {fid_score:.2f}")

        # Calculate statistics
        mean_fid = np.mean(all_fids)
        std_fid = np.std(all_fids)

        print(f"Final FID: {mean_fid:.2f} ± {std_fid:.2f}")

        # Save results
        final_fid_results = {
            "final_fid_mean": mean_fid,
            "final_fid_std": std_fid,
            "final_fid_runs": all_fids,
            "num_samples": num_samples,
            "num_runs": num_runs,
            "base_seed": seed
        }

        results_path = os.path.join(self.output_dir, "results.json")
        if os.path.exists(results_path):
            with open(results_path, "r") as f:
                results = json.load(f)

            results.update(final_fid_results)

            with open(results_path, "w") as f:
                json.dump(results, f, indent=4)
        else:
            with open(results_path, "w") as f:
                json.dump(final_fid_results, f, indent=4)

        return mean_fid, std_fid, all_fids

    def train(self, epochs):
        """
        Train the model for the specified number of epochs.

        Args:
            epochs (int): Number of epochs to train for

        Returns:
            dict: Dictionary with training results
        """
        start_time = time.time()

        for epoch in range(epochs):
            # Train for one epoch
            train_loss, recon_loss, kl_loss = self.train_epoch(epoch)

            # Validate
            val_loss, val_recon, val_kl = self.validate(epoch)

            # Update learning rate if scheduler is provided
            if self.scheduler is not None:
                self.scheduler.step(val_loss)

            # Track losses
            self.train_losses.append(train_loss)
            self.recon_losses.append(recon_loss)
            self.kl_losses.append(kl_loss)

            # Compute FID score periodically
            if (epoch + 1) % self.config.get("fid_interval", 5) == 0:
                fid_score = self.compute_fid()
                self.fid_scores.append(fid_score)

                print(f"Epoch {epoch+1}/{epochs}, FID: {fid_score:.2f}")

                # Check if this is the best model so far
                if fid_score < self.best_fid:
                    self.best_fid = fid_score
                    self.best_epoch = epoch + 1
                    self.save_checkpoint(epoch, is_best=True)
                    print(f"New best model with FID: {fid_score:.2f}")

            # Save samples and reconstructions periodically
            if (epoch + 1) % self.config.get("plot_interval", 10) == 0:
                self.save_samples(epoch)
                self.save_reconstructions(epoch)
                self.plot_losses()
                self.plot_fid_scores()

            # Save checkpoint
            self.save_checkpoint(epoch)

            # Print progress
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Calculate training time
        end_time = time.time()
        training_time = {
            "seconds": end_time - start_time,
            "formatted": time.strftime("%H:%M:%S", time.gmtime(end_time - start_time))
        }

        # Save final samples and reconstructions
        self.save_samples(epochs-1, num_samples=16, is_final=True)
        self.save_reconstructions(epochs-1, num_samples=8, is_final=True)

        # Plot final losses and FID scores
        self.plot_losses()
        self.plot_fid_scores()

        # Save results
        self.save_results(training_time)

        # Compute final FID score
        mean_fid, std_fid, all_fids = self.compute_final_fid()

        return {
            "model": self.model,
            "train_losses": self.train_losses,
            "recon_losses": self.recon_losses,
            "kl_losses": self.kl_losses,
            "fid_scores": self.fid_scores,
            "best_fid": self.best_fid,
            "best_epoch": self.best_epoch,
            "final_fid_mean": mean_fid,
            "final_fid_std": std_fid,
            "final_fid_runs": all_fids,
            "training_time": training_time
        }


def get_dataloaders(batch_size=128, use_augmentation=False, split_ratio=0.9, seed=42):
    """
    Get dataloaders for the BloodMNIST dataset.

    Args:
        batch_size (int): Batch size for dataloaders
        use_augmentation (bool): Whether to use data augmentation
        split_ratio (float): Ratio of training to validation data
        seed (int): Random seed for reproducibility

    Returns:
        tuple: (train_loader, val_loader, full_loader)
    """
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Define transformations
    if use_augmentation:
        # With data augmentation
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # Normalize to [-1, 1]
        ])
    else:
        # Without data augmentation
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # Normalize to [-1, 1]
        ])

    # Load datasets
    train_dataset = BloodMNIST(root="./data", split="train", transform=transform, download=True)
    val_dataset = BloodMNIST(root="./data", split="val", transform=transform, download=True)
    test_dataset = BloodMNIST(root="./data", split="test", transform=transform, download=True)

    # Combine all datasets for training (as per project requirements)
    full_dataset = ConcatDataset([train_dataset, val_dataset, test_dataset])

    # Split into train and validation sets
    train_size = int(split_ratio * len(full_dataset))
    val_size = len(full_dataset) - train_size

    train_subset, val_subset = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    full_loader = DataLoader(
        full_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    return train_loader, val_loader, full_loader


def train_vae(config):
    """
    Train a VAE model with the specified configuration.

    Args:
        config (dict): Configuration dictionary with hyperparameters

    Returns:
        dict: Dictionary with training results
    """
    # Set random seed for reproducibility
    seed = config.get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get dataloaders
    train_loader, val_loader, full_loader = get_dataloaders(
        batch_size=config.get("batch_size", 128),
        use_augmentation=config.get("use_augmentation", True),
        split_ratio=config.get("split_ratio", 0.9),
        seed=seed
    )

    # Create model
    model = ConvVAE(
        in_channels=config.get("in_channels", 3),
        latent_dim=config.get("latent_dim", 128),
        hidden_dims=config.get("hidden_dims", [32, 64, 128, 256]),
        img_size=config.get("img_size", 28),
        beta=config.get("beta", 1.0)
    ).to(device)

    # Create optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.get("learning_rate", 3e-4),
        betas=config.get("betas", (0.9, 0.999))
    )

    # Create scheduler
    scheduler = None
    if config.get("use_scheduler", True):
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            min_lr=config.get("min_lr", 1e-5),
            verbose=True
        )

    # Create trainer
    trainer = VAETrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader if not config.get("use_val_for_fid", False) else full_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        config=config,
        output_dir=config.get("output_dir", "./results")
    )

    # Train model
    results = trainer.train(config.get("epochs", 100))

    return results


if __name__ == "__main__":
    # Example configuration
    config = {
        "model_name": "ConvVAE",
        "output_dir": "./vae_results",

        # Model parameters
        "in_channels": 3,           # RGB images
        "latent_dim": 128,          # Dimension of latent space
        "hidden_dims": [32, 64, 128, 256],  # Hidden dimensions for conv layers
        "img_size": 28,             # BloodMNIST images are 28x28
        "beta": 0.5,                # Weight for KL divergence term

        # Training parameters
        "batch_size": 128,
        "learning_rate": 3e-4,
        "epochs": 200,
        "use_scheduler": True,
        "min_lr": 1e-5,
        "use_augmentation": True,

        # Evaluation parameters
        "fid_interval": 5,          # Calculate FID every 5 epochs
        "plot_interval": 10,        # Update plots every 10 epochs
        "use_val_for_fid": False,   # Use full dataset for FID calculation

        # Reproducibility
        "seed": 42
    }

    # Train the model
    results = train_vae(config)

    print(f"Training complete. Best FID: {results['best_fid']:.2f} at epoch {results['best_epoch']}")
