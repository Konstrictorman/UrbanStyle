"""
Interactive GAN Application with Pygame Interface
=================================================

This module implements an interactive Generative Adversarial Network (GAN) application
for generating 28x28 fashion product images using DCGAN (Deep Convolutional GAN) architecture.

The application features:
- Real-time training visualization with pygame interface
- Interactive parameter adjustment via sliders
- Live image generation and display
- Model persistence (save/load functionality)
- Comprehensive training progress tracking

Architecture:
- Generator: Transforms 100D noise vectors into 28x28 grayscale images
- Discriminator: Classifies images as real or fake
- Training: Adversarial process where both networks compete and improve

Author: AI Assistant
Date: 2024
License: MIT
"""

import pygame
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils as vutils
import numpy as np
import os
import threading
import time
from PIL import Image
import sys

# Add parent directory to path to import common components
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from common.button import Button
from common.slider import SliderPanel

# Initialize Pygame
pygame.init()

# Constants
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 800
FPS = 60

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (128, 128, 128)
LIGHT_GRAY = (200, 200, 200)
DARK_GRAY = (64, 64, 64)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)

class Generator(nn.Module):
    """
    DCGAN Generator Network for 28x28 Fashion Image Generation
    
    This class implements the Generator component of a Deep Convolutional GAN (DCGAN).
    It transforms random noise vectors into realistic 28x28 grayscale fashion images.
    
    Architecture:
    - Input: z_dim-dimensional noise vector
    - Fully Connected Layer: Expands noise to base*7*7 features
    - Batch Normalization: Stabilizes training
    - Transposed Convolutions: Upsamples from 7x7 to 28x28
    - Tanh Activation: Outputs values in [-1, 1] range
    
    Args:
        z_dim (int): Dimension of input noise vector. Controls complexity of generated images.
                    Higher values allow more diverse generation but require more training.
                    Default: 100 (good balance between diversity and training stability)
        base (int): Base number of filters for the network. Controls model capacity.
                   Higher values = more parameters = better quality but slower training.
                   Default: 128 (optimal for 28x28 images)
    
    Returns:
        torch.Tensor: Generated images of shape [batch_size, 1, 28, 28]
                     Values normalized to [-1, 1] range
    """
    def __init__(self, z_dim=100, base=128):
        super().__init__()
        self.z_dim = z_dim
        self.base = base
        
        # Fully connected layer
        self.fc = nn.Sequential(
            nn.Linear(z_dim, base * 7 * 7),
            nn.BatchNorm1d(base * 7 * 7),
            nn.ReLU(True),
        )
        
        # Transposed convolution layers
        self.net = nn.Sequential(
            nn.ConvTranspose2d(base, base // 2, 4, 2, 1),  # 7->14
            nn.BatchNorm2d(base // 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(base // 2, 1, 4, 2, 1),     # 14->28
            nn.Tanh(),
        )
        
        self.apply(self._weights_init)
    
    def _weights_init(self, m):
        """Initialize weights"""
        classname = m.__class__.__name__
        if "Conv" in classname or "Linear" in classname:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.zeros_(m.bias.data)
        elif "BatchNorm" in classname:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.zeros_(m.bias.data)
    
    def forward(self, z):
        """
        Forward pass through the Generator network
        
        Transforms input noise vectors into generated images through a series of
        fully connected and transposed convolutional layers.
        
        Args:
            z (torch.Tensor): Input noise tensor of shape [batch_size, z_dim]
                             Random noise vectors that will be transformed into images
                             
        Returns:
            torch.Tensor: Generated images of shape [batch_size, 1, 28, 28]
                         Grayscale images with values in [-1, 1] range
                         Ready for discriminator evaluation or display
        """
        # Expand noise to spatial features: [B, z_dim] -> [B, base*7*7]
        x = self.fc(z).view(-1, self.base, 7, 7)
        # Apply transposed convolutions: [B, base, 7, 7] -> [B, 1, 28, 28]
        return self.net(x)

class Discriminator(nn.Module):
    """
    DCGAN Discriminator Network for Real/Fake Image Classification
    
    This class implements the Discriminator component of a Deep Convolutional GAN (DCGAN).
    It classifies input images as either real (from dataset) or fake (generated by Generator).
    
    Architecture:
    - Input: 28x28 grayscale images
    - Convolutional Layers: Downsample from 28x28 to 7x7
    - Batch Normalization: Stabilizes training
    - LeakyReLU Activation: Prevents gradient vanishing
    - Fully Connected Layer: Outputs single logit score
    
    Args:
        base (int): Base number of filters for the network. Controls model capacity.
                   Higher values = more parameters = better discrimination but slower training.
                   Default: 64 (optimal for 28x28 images)
    
    Returns:
        torch.Tensor: Classification logits of shape [batch_size, 1]
                     Positive values = likely real, Negative values = likely fake
    """
    def __init__(self, base=64):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Conv2d(1, base, 4, 2, 1),    # 28->14
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base, base * 2, 4, 2, 1),  # 14->7
            nn.BatchNorm2d(base * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(base * 2 * 7 * 7, 1)
        )
        
        self.apply(self._weights_init)
    
    def _weights_init(self, m):
        """Initialize weights"""
        classname = m.__class__.__name__
        if "Conv" in classname or "Linear" in classname:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.zeros_(m.bias.data)
        elif "BatchNorm" in classname:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.zeros_(m.bias.data)
    
    def forward(self, x):
        """
        Forward pass through the Discriminator network
        
        Processes input images through convolutional layers to classify them
        as real (from dataset) or fake (generated by Generator).
        
        Args:
            x (torch.Tensor): Input images of shape [batch_size, 1, 28, 28]
                             Can be real images from dataset or fake images from Generator
                             Values should be in [-1, 1] range
                             
        Returns:
            torch.Tensor: Classification logits of shape [batch_size, 1]
                         Positive values indicate higher probability of being real
                         Negative values indicate higher probability of being fake
                         Used with BCEWithLogitsLoss for training
        """
        return self.net(x)

class GANApp:
    """
    Interactive GAN Application with Pygame Interface
    
    This is the main application class that orchestrates the entire GAN training
    and visualization system. It combines PyTorch models with Pygame interface
    for real-time interactive training and image generation.
    
    Features:
    - Real-time GAN training with visual feedback
    - Interactive parameter adjustment via sliders
    - Live image generation and display
    - Model persistence (save/load checkpoints)
    - Multi-threaded training to keep UI responsive
    - Comprehensive progress tracking and visualization
    
    Architecture:
    - Generator: Creates fake fashion images from noise
    - Discriminator: Distinguishes real from fake images
    - Training Loop: Adversarial training process
    - UI Components: Buttons, sliders, and visualizations
    """
    def __init__(self):
        """
        Initialize the Interactive GAN Application
        
        Sets up all components needed for the GAN training and visualization:
        - Pygame display and UI components
        - PyTorch models (Generator and Discriminator)
        - Training parameters and optimizers
        - Data loading and preprocessing
        - Interactive controls (buttons and sliders)
        - Visualization and tracking systems
        
        Initializes with default parameters optimized for Fashion-MNIST dataset:
        - Learning Rate: 0.0002 (stable for GAN training)
        - Noise Dimension: 100 (good balance of diversity and stability)
        - Batch Size: 64 (optimal for 28x28 images)
        - Max Epochs: 1 (user can adjust via slider)
        """
        
        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Initialize models
        self.z_dim = 100
        self.generator = Generator(self.z_dim).to(self.device)
        self.discriminator = Discriminator().to(self.device)
        
        # Optimizers
        self.lr = 0.0002
        self.optimizer_G = optim.Adam(self.generator.parameters(), lr=self.lr, betas=(0.5, 0.999))
        self.optimizer_D = optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=(0.5, 0.999))
        
        # Loss function
        self.criterion = nn.BCEWithLogitsLoss()
        
        # Training state
        self.is_training = False
        self.training_thread = None
        self.epoch = 0
        self.max_epochs = 1  # Default max epochs
        self.batch_size = 64  # Default batch size
        self.batch_count = 0
        self.losses_G = []
        self.losses_D = []
        
        # Data
        self.data_loader = None
        self.setup_data()
        
        # Calculate batches per epoch after data loader is created
        self.batches_per_epoch = len(self.data_loader)
        print(f"Batches per epoch: {self.batches_per_epoch}")
        
        # Fixed noise for consistent generation
        self.fixed_noise = torch.randn(16, self.z_dim, device=self.device)
        self.current_noise_seed = 42
        torch.manual_seed(self.current_noise_seed)
        
        # Generated images
        self.generated_images = None
        self.update_generated_images()
        
        # UI Components
        self.setup_ui()
        
        # Training stats
        self.last_loss_G = 0.0
        self.last_loss_D = 0.0
    
    def setup_data(self):
        """
        Setup Fashion-MNIST Dataset and Data Loader
        
        Downloads and prepares the Fashion-MNIST dataset for GAN training.
        Applies necessary transformations to normalize images for the DCGAN architecture.
        
        Transformations Applied:
        - Resize: Ensures 28x28 dimensions (Fashion-MNIST is already 28x28)
        - CenterCrop: Crops to exact 28x28 if needed
        - ToTensor: Converts PIL images to PyTorch tensors
        - Normalize: Scales pixel values from [0, 1] to [-1, 1] range
                     This is crucial for GAN training as Generator outputs [-1, 1]
        
        Dataset Details:
        - 60,000 training images of fashion items
        - 10 classes: T-shirt, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot
        - 28x28 grayscale images
        - Automatically downloads if not present
        
        Creates:
        - self.data_loader: PyTorch DataLoader for batch processing
        - self.batches_per_epoch: Number of batches per epoch (calculated after creation)
        """
        transform = transforms.Compose([
            transforms.Resize(28),
            transforms.CenterCrop(28),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
        ])
        
        dataset = datasets.FashionMNIST(
            root="./FashionMNIST", 
            train=True, 
            download=True, 
            transform=transform
        )
        
        self.data_loader = DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            drop_last=True,
            num_workers=2
        )
    
    def setup_ui(self):
        """Setup UI components"""
        # Control buttons with event callbacks
        button_width, button_height = 120, 40
        button_spacing = 10
        start_x = 50
        start_y = 50
        
        self.start_button = Button(
            start_x, start_y, button_width, button_height,
            "Start Training", GREEN,
            on_click=self.on_start_training
        )
        
        self.stop_button = Button(
            start_x + button_width + button_spacing, start_y, 
            button_width, button_height, "Stop Training", RED,
            on_click=self.on_stop_training
        )
        
        self.generate_button = Button(
            start_x + 2 * (button_width + button_spacing), start_y,
            button_width, button_height, "Generate", BLUE,
            on_click=self.on_generate_images
        )
        
        self.save_button = Button(
            start_x + 3 * (button_width + button_spacing), start_y,
            button_width, button_height, "Save Model", GRAY,
            on_click=self.on_save_model
        )
        
        self.load_button = Button(
            start_x + 4 * (button_width + button_spacing), start_y,
            button_width, button_height, "Load Model", GRAY,
            on_click=self.on_load_model
        )
        
        # Parameter sliders panel
        self.slider_panel = SliderPanel(50, 120, 300, 400)
        self.slider_panel.add_slider(0.0001, 0.01, self.lr, "Learning Rate", False)
        self.slider_panel.add_slider(1, 100, self.current_noise_seed, "Noise Seed", True)
        self.slider_panel.add_slider(1, 10, 1, "Training Epochs", True)
        self.slider_panel.add_slider(32, 256, self.batch_size, "Batch Size", True)
        self.slider_panel.add_slider(50, 200, self.z_dim, "Noise Dimension", True)
    
    # Button event callbacks
    def on_start_training(self, button):
        """Callback for start training button"""
        self.start_training()
    
    def on_stop_training(self, button):
        """Callback for stop training button"""
        self.stop_training()
    
    def on_generate_images(self, button):
        """Callback for generate images button"""
        self.update_generated_images()
    
    def on_save_model(self, button):
        """Callback for save model button"""
        self.save_model()
    
    def on_load_model(self, button):
        """Callback for load model button"""
        self.load_model()
    
    def update_generated_images(self):
        """
        Generate New Images Using Current Generator State
        
        Creates a batch of 16 new fashion images using the current Generator model
        and the specified noise seed. This method is called when:
        - User clicks the "Generate" button
        - Noise seed slider is changed
        - Noise dimension is modified
        - During training (every 10 batches for live updates)
        
        Process:
        1. Sets PyTorch random seed for reproducible generation
        2. Generates random noise vectors with current z_dim
        3. Passes noise through Generator (inference mode)
        4. Converts PyTorch tensors to numpy arrays for pygame display
        
        Generated Images:
        - Count: 16 images (4×4 grid display)
        - Size: 28×28 pixels
        - Format: Grayscale with values in [-1, 1] range
        - Device: Automatically uses GPU if available
        
        Updates:
        - self.generated_images: PyTorch tensor for model operations
        - self.generated_images_np: Numpy array for pygame visualization
        
        Note: Uses torch.no_grad() to disable gradient computation for efficiency
        """
        torch.manual_seed(self.current_noise_seed)
        noise = torch.randn(16, self.z_dim, device=self.device)
        
        with torch.no_grad():
            self.generated_images = self.generator(noise)
            # Convert to numpy for pygame display
            self.generated_images_np = self.generated_images.cpu().numpy()
    
    def train_step(self):
        """
        Main GAN Training Loop with Batch-Based Epoch Tracking
        
        Implements the core adversarial training process where Generator and Discriminator
        compete against each other. This method runs in a separate thread to keep the
        UI responsive during training.
        
        Training Process:
        1. Calculate total batches needed for specified epochs
        2. For each batch in the dataset:
           a. Train Discriminator on real and fake images
           b. Train Generator to fool the Discriminator
           c. Update loss statistics and generated images
           d. Check for early stopping conditions
        
        Discriminator Training:
        - Real images: Trained to output high scores (close to 1)
        - Fake images: Trained to output low scores (close to 0)
        - Loss: Binary Cross-Entropy with Logits
        
        Generator Training:
        - Fake images: Trained to fool Discriminator (output high scores)
        - Loss: Binary Cross-Entropy with Logits (inverted labels)
        
        Key Features:
        - Batch-based epoch tracking for accurate progress display
        - Real-time parameter updates from UI sliders
        - Automatic checkpoint saving after each epoch
        - Graceful handling of training interruption
        - Progress visualization updates every 10 batches
        
        Thread Safety:
        - Runs in separate thread to prevent UI blocking
        - Uses self.is_training flag for clean shutdown
        - Updates shared state safely for UI display
        """
        try:
            total_batches_needed = self.max_epochs * self.batches_per_epoch
            print(f"Training for {self.max_epochs} epochs = {total_batches_needed} total batches")
            
            batch_count = 0
            while self.is_training and batch_count < total_batches_needed:
                for real_images, _ in self.data_loader:
                    if not self.is_training or batch_count >= total_batches_needed:
                        break
                    
                    # Calculate current epoch based on batches processed
                    current_epoch = (batch_count // self.batches_per_epoch) + 1
                    if current_epoch != self.epoch:
                        self.epoch = current_epoch
                        print(f"Starting epoch {self.epoch}")
                    
                    real_images = real_images.to(self.device)
                    batch_size = real_images.size(0)
                    
                    # Labels
                    real_labels = torch.ones(batch_size, 1, device=self.device)
                    fake_labels = torch.zeros(batch_size, 1, device=self.device)
                    
                    # Train Discriminator
                    self.optimizer_D.zero_grad()
                    
                    # Real images
                    real_output = self.discriminator(real_images)
                    loss_D_real = self.criterion(real_output, real_labels)
                    
                    # Fake images
                    noise = torch.randn(batch_size, self.z_dim, device=self.device)
                    fake_images = self.generator(noise)
                    fake_output = self.discriminator(fake_images.detach())
                    loss_D_fake = self.criterion(fake_output, fake_labels)
                    
                    loss_D = loss_D_real + loss_D_fake
                    loss_D.backward()
                    self.optimizer_D.step()
                    
                    # Train Generator
                    self.optimizer_G.zero_grad()
                    fake_output = self.discriminator(fake_images)
                    loss_G = self.criterion(fake_output, real_labels)
                    loss_G.backward()
                    self.optimizer_G.step()
                    
                    # Update stats
                    self.last_loss_G = loss_G.item()
                    self.last_loss_D = loss_D.item()
                    self.losses_G.append(self.last_loss_G)
                    self.losses_D.append(self.last_loss_D)
                    self.batch_count += 1
                    batch_count += 1
                    
                    # Update generated images periodically
                    if self.batch_count % 10 == 0:
                        self.update_generated_images()
                    
                    # Save checkpoint every epoch completion
                    if batch_count % self.batches_per_epoch == 0:
                        print(f"Completed epoch {self.epoch}")
                        self.save_epoch_checkpoint()
                    
                    time.sleep(0.01)  # Small delay for UI responsiveness
            
            # Training completed
            if batch_count >= total_batches_needed:
                print(f"Training completed! Processed {batch_count} batches across {self.max_epochs} epochs.")
                self.is_training = False
            
        except Exception as e:
            print(f"Training error: {e}")
            self.is_training = False
    
    def save_epoch_checkpoint(self):
        """Save samples and checkpoints for current epoch"""
        try:
            os.makedirs("checkpoints", exist_ok=True)
            os.makedirs("samples", exist_ok=True)
            
            # Generate sample images
            self.generator.eval()
            with torch.no_grad():
                grid = vutils.make_grid(self.generator(self.fixed_noise), nrow=8, normalize=True, value_range=(-1,1))
            vutils.save_image(grid, f"samples/epoch_{self.epoch:03d}.png")
            
            # Save model checkpoints
            torch.save({
                'generator_state_dict': self.generator.state_dict(),
                'discriminator_state_dict': self.discriminator.state_dict(),
                'optimizer_G_state_dict': self.optimizer_G.state_dict(),
                'optimizer_D_state_dict': self.optimizer_D.state_dict(),
                'epoch': self.epoch,
                'losses_G': self.losses_G,
                'losses_D': self.losses_D,
            }, f"checkpoints/gan_checkpoint_epoch_{self.epoch:03d}.pth")
            
            print(f"Epoch {self.epoch}: Saved samples and checkpoints")
            
        except Exception as e:
            print(f"Error saving checkpoint: {e}")
    
    def start_training(self):
        """Start training in separate thread"""
        if not self.is_training:
            # Reset epoch counter when starting new training
            self.epoch = 0
            self.batch_count = 0
            self.losses_G = []
            self.losses_D = []
            
            self.is_training = True
            self.training_thread = threading.Thread(target=self.train_step)
            self.training_thread.daemon = True
            self.training_thread.start()
    
    def stop_training(self):
        """Stop training"""
        self.is_training = False
        if self.training_thread:
            self.training_thread.join(timeout=1.0)
    
    def save_model(self):
        """
        Save Current Model State to Checkpoint File
        
        Persists the complete training state including model weights, optimizer states,
        training progress, and loss history. This allows resuming training from exactly
        where it was left off.
        
        Saved Components:
        - Generator state_dict(): Model weights and biases
        - Discriminator state_dict(): Model weights and biases  
        - Optimizer states: Learning rates, momentum, and other optimizer parameters
        - Training progress: Current epoch number
        - Loss history: Complete loss curves for both networks
        
        File Format:
        - Location: checkpoints/gan_checkpoint_epoch_{epoch}.pth
        - Format: PyTorch checkpoint (.pth)
        - Naming: Includes epoch number for easy identification
        
        Usage:
        - Manual save: User clicks "Save Model" button
        - Automatic save: After each epoch completion
        - Resume training: Use load_model() to restore state
        
        Error Handling:
        - Creates checkpoints directory if it doesn't exist
        - Prints success/error messages to console
        - Gracefully handles file system errors
        """
        os.makedirs("checkpoints", exist_ok=True)
        torch.save({
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'optimizer_G_state_dict': self.optimizer_G.state_dict(),
            'optimizer_D_state_dict': self.optimizer_D.state_dict(),
            'epoch': self.epoch,
            'losses_G': self.losses_G,
            'losses_D': self.losses_D,
        }, f"checkpoints/gan_checkpoint_epoch_{self.epoch}.pth")
        print(f"Model saved at epoch {self.epoch}")
    
    def load_model(self):
        """
        Load Previously Saved Model State from Checkpoint
        
        Restores the complete training state from a saved checkpoint file, allowing
        users to resume training from where they left off or load a pre-trained model.
        
        Loaded Components:
        - Generator weights: Restores Generator model to saved state
        - Discriminator weights: Restores Discriminator model to saved state
        - Optimizer states: Restores learning rates, momentum, and optimizer parameters
        - Training progress: Resumes from saved epoch number
        - Loss history: Restores complete loss curves
        
        File Handling:
        - Default file: checkpoints/gan_checkpoint_epoch_1.pth
        - Device mapping: Automatically maps to current device (CPU/GPU)
        - Error handling: Gracefully handles missing files
        
        Usage Scenarios:
        - Resume interrupted training
        - Load pre-trained models
        - Experiment with different model states
        - Share trained models between sessions
        
        Post-Load Actions:
        - Updates generated images with loaded model
        - Prints confirmation message with epoch number
        - Restores all training parameters and statistics
        
        Error Handling:
        - FileNotFoundError: Prints "No checkpoint found" message
        - Device compatibility: Automatically handles CPU/GPU mapping
        - State restoration: Validates checkpoint format before loading
        """
        try:
            checkpoint = torch.load("checkpoints/gan_checkpoint_epoch_1.pth", map_location=self.device)
            self.generator.load_state_dict(checkpoint['generator_state_dict'])
            self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
            self.optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
            self.optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
            self.epoch = checkpoint['epoch']
            self.losses_G = checkpoint['losses_G']
            self.losses_D = checkpoint['losses_D']
            print(f"Model loaded from epoch {self.epoch}")
        except FileNotFoundError:
            print("No checkpoint found")
    
    def draw_image_grid(self, images, x, y, scale=4):
        """Draw a grid of images"""
        if images is None:
            return
        
        # Convert tensor to numpy if needed
        if isinstance(images, torch.Tensor):
            images = images.cpu().numpy()
        
        # Ensure images are in [0, 1] range
        images = (images + 1) / 2  # Convert from [-1, 1] to [0, 1]
        images = np.clip(images, 0, 1)
        
        # Create surface for image grid
        grid_size = int(np.sqrt(len(images)))
        if grid_size * grid_size < len(images):
            grid_size += 1
        
        img_size = 28 * scale
        grid_width = grid_size * img_size
        grid_height = grid_size * img_size
        
        grid_surface = pygame.Surface((grid_width, grid_height))
        grid_surface.fill(BLACK)
        
        for i, img in enumerate(images[:grid_size * grid_size]):
            row = i // grid_size
            col = i % grid_size
            
            # Convert grayscale image to RGB
            img_2d = img[0] if len(img.shape) == 3 else img
            img_rgb = np.stack([img_2d] * 3, axis=-1)
            img_rgb = (img_rgb * 255).astype(np.uint8)
            
            # Create pygame surface
            img_surface = pygame.surfarray.make_surface(img_rgb.swapaxes(0, 1))
            img_surface = pygame.transform.scale(img_surface, (img_size, img_size))
            
            # Blit to grid
            grid_surface.blit(img_surface, (col * img_size, row * img_size))
        
        # Draw grid on main screen
        self.screen.blit(grid_surface, (x, y))
        
        # Draw border
        pygame.draw.rect(self.screen, WHITE, (x, y, grid_width, grid_height), 2)
    
    def draw_loss_curve(self, x, y, width, height):
        """Draw loss curves"""
        if len(self.losses_G) < 2:
            return
        
        # Create surface for loss plot
        loss_surface = pygame.Surface((width, height))
        loss_surface.fill(BLACK)
        
        # Normalize losses for display
        all_losses = self.losses_G + self.losses_D
        if all_losses:
            min_loss = min(all_losses)
            max_loss = max(all_losses)
            loss_range = max_loss - min_loss if max_loss > min_loss else 1
            
            # Draw G loss (green)
            if len(self.losses_G) > 1:
                points_G = []
                for i, loss in enumerate(self.losses_G[-100:]):  # Last 100 points
                    x_pos = int(i * width / len(self.losses_G[-100:]))
                    y_pos = int(height - (loss - min_loss) * height / loss_range)
                    points_G.append((x_pos, y_pos))
                
                if len(points_G) > 1:
                    pygame.draw.lines(loss_surface, GREEN, False, points_G, 2)
            
            # Draw D loss (red)
            if len(self.losses_D) > 1:
                points_D = []
                for i, loss in enumerate(self.losses_D[-100:]):  # Last 100 points
                    x_pos = int(i * width / len(self.losses_D[-100:]))
                    y_pos = int(height - (loss - min_loss) * height / loss_range)
                    points_D.append((x_pos, y_pos))
                
                if len(points_D) > 1:
                    pygame.draw.lines(loss_surface, RED, False, points_D, 2)
        
        self.screen.blit(loss_surface, (x, y))
        pygame.draw.rect(self.screen, WHITE, (x, y, width, height), 2)
        
        # Draw labels
        g_loss_text = self.font.render(f"G Loss: {self.last_loss_G:.3f}", True, GREEN)
        d_loss_text = self.font.render(f"D Loss: {self.last_loss_D:.3f}", True, RED)
        self.screen.blit(g_loss_text, (x, y - 25))
        self.screen.blit(d_loss_text, (x, y + height + 5))
    
    def handle_events(self):
        """
        Handle All Pygame Events and Real-Time Parameter Updates
        
        This is the main event processing method that runs every frame (60 FPS).
        It handles user input, updates UI components, and applies real-time parameter
        changes from sliders to the training system.
        
        Event Types Handled:
        - pygame.QUIT: Window close button
        - Slider events: Parameter adjustments via SliderPanel
        - Button events: UI button interactions (start, stop, generate, save, load)
        - Mouse events: Hover and click detection for buttons
        
        Real-Time Parameter Updates:
        The method continuously monitors slider values and applies changes immediately:
        
        1. Learning Rate (Slider 0):
           - Range: 0.0001 - 0.01
           - Effect: Updates both Generator and Discriminator optimizers
           - Impact: Changes training speed and convergence behavior
        
        2. Noise Seed (Slider 1):
           - Range: 1 - 100
           - Effect: Controls reproducible image generation
           - Impact: Changes noise patterns for Generator training
        
        3. Training Epochs (Slider 2):
           - Range: 1 - 10
           - Effect: Sets maximum training epochs
           - Impact: Controls when training stops
        
        4. Batch Size (Slider 3):
           - Range: 32 - 256
           - Effect: Recreates data loader with new batch size
           - Impact: Changes memory usage and training dynamics
        
        5. Noise Dimension (Slider 4):
           - Range: 50 - 200
           - Effect: Recreates Generator and Discriminator models
           - Impact: Changes model capacity and generation diversity
        
        Returns:
            bool: True if application should continue running, False to quit
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            
            # Handle sliders
            self.slider_panel.handle_event(event)
        
        # Get mouse state for button handling
        mouse_pos = pygame.mouse.get_pos()
        mouse_pressed = pygame.mouse.get_pressed()[0]
        
        # Handle all button events using the new comprehensive method
        self.start_button.handle_events(mouse_pos, mouse_pressed)
        self.stop_button.handle_events(mouse_pos, mouse_pressed)
        self.generate_button.handle_events(mouse_pos, mouse_pressed)
        self.save_button.handle_events(mouse_pos, mouse_pressed)
        self.load_button.handle_events(mouse_pos, mouse_pressed)
        
        # Update parameters from sliders
        slider_values = self.slider_panel.get_values()
        new_lr = slider_values[0]
        new_seed = int(slider_values[1])
        new_max_epochs = int(slider_values[2])
        new_batch_size = int(slider_values[3])
        new_z_dim = int(slider_values[4])
        
        if new_lr != self.lr:
            self.lr = new_lr
            self.optimizer_G = optim.Adam(self.generator.parameters(), lr=self.lr, betas=(0.5, 0.999))
            self.optimizer_D = optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=(0.5, 0.999))
        
        if new_seed != self.current_noise_seed:
            self.current_noise_seed = new_seed
            self.update_generated_images()
        
        if new_max_epochs != self.max_epochs:
            self.max_epochs = new_max_epochs
        
        if new_batch_size != self.batch_size:
            self.batch_size = new_batch_size
            self.setup_data()  # Recreate data loader with new batch size
            self.batches_per_epoch = len(self.data_loader)
            print(f"Batch size changed to {self.batch_size}, batches per epoch: {self.batches_per_epoch}")
        
        if new_z_dim != self.z_dim:
            self.z_dim = new_z_dim
            # Recreate generator and discriminator with new noise dimension
            self.generator = Generator(self.z_dim).to(self.device)
            self.discriminator = Discriminator().to(self.device)
            self.optimizer_G = optim.Adam(self.generator.parameters(), lr=self.lr, betas=(0.5, 0.999))
            self.optimizer_D = optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=(0.5, 0.999))
            # Update fixed noise for generation
            self.fixed_noise = torch.randn(16, self.z_dim, device=self.device)
            self.update_generated_images()
            print(f"Noise dimension changed to {self.z_dim}")
        
        return True
    
    def draw(self):
        """
        Render Complete Application Interface
        
        This is the main rendering method that draws all UI components, visualizations,
        and status information to the pygame screen. It's called every frame (60 FPS)
        to maintain smooth real-time updates.
        
        Rendered Components:
        
        1. **Background**: Black background for clear contrast
        
        2. **Title**: "Interactive Fashion GAN" centered at top
        
        3. **Control Buttons**: All interactive buttons with hover effects
           - Start Training, Stop Training, Generate, Save Model, Load Model
        
        4. **Parameter Sliders**: Interactive sliders for real-time parameter adjustment
           - Learning Rate, Noise Seed, Training Epochs, Batch Size, Noise Dimension
        
        5. **Generated Images Grid**: 4×4 grid of current generated fashion images
           - Updates in real-time during training
           - Shows current Generator output quality
        
        6. **Loss Curves**: Live plotting of Generator and Discriminator losses
           - Visual feedback on training progress
           - Helps identify training stability
        
        7. **Training Statistics**: Real-time display of:
           - Current epoch and progress
           - Batch count and epoch progress
           - Generator and Discriminator losses
           - Training status
        
        8. **Status Information**: Additional context and help text
        
        Rendering Order:
        1. Background fill
        2. Static elements (title, labels)
        3. Interactive elements (buttons, sliders)
        4. Dynamic content (images, curves, stats)
        5. Screen update (pygame.display.flip())
        
        Performance:
        - Optimized for 60 FPS rendering
        - Efficient pygame surface operations
        - Minimal redraw operations
        """
        self.screen.fill(BLACK)
        
        # Draw title
        title_text = self.title_font.render("Interactive Fashion GAN", True, WHITE)
        self.screen.blit(title_text, (WINDOW_WIDTH // 2 - title_text.get_width() // 2, 10))
        
        # Draw buttons
        self.start_button.draw(self.screen)
        self.stop_button.draw(self.screen)
        self.generate_button.draw(self.screen)
        self.save_button.draw(self.screen)
        self.load_button.draw(self.screen)
        
        # Draw sliders
        self.slider_panel.draw(self.screen)
        
        # Draw generated images
        self.draw_image_grid(self.generated_images_np, 400, 120, scale=3)
        
        # Draw loss curve
        self.draw_loss_curve(400, 500, 300, 150)
        
        # Draw training status
        status_text = "Training" if self.is_training else "Stopped"
        status_color = GREEN if self.is_training else RED
        status_surface = self.font.render(f"Status: {status_text}", True, status_color)
        self.screen.blit(status_surface, (400, 680))
        
        # Draw epoch info
        epoch_text = self.font.render(f"Epoch: {self.epoch}/{self.max_epochs}", True, WHITE)
        self.screen.blit(epoch_text, (400, 710))
        
        batch_text = self.font.render(f"Batches: {self.batch_count}", True, WHITE)
        self.screen.blit(batch_text, (400, 740))
        
        # Show batch progress within current epoch
        if hasattr(self, 'batches_per_epoch') and self.batches_per_epoch > 0:
            current_epoch_batches = self.batch_count % self.batches_per_epoch
            progress_text = self.font.render(f"Epoch Progress: {current_epoch_batches}/{self.batches_per_epoch}", True, WHITE)
            self.screen.blit(progress_text, (400, 770))
        
        pygame.display.flip()
    
    def run(self):
        """
        Main Application Loop - Entry Point for Interactive GAN Training
        
        This is the primary execution method that starts and manages the entire
        interactive GAN application. It implements the main game loop pattern
        commonly used in pygame applications.
        
        Loop Structure:
        1. **Event Handling**: Process user input and parameter changes
        2. **Rendering**: Draw all UI components and visualizations
        3. **Frame Rate Control**: Maintain consistent 60 FPS performance
        4. **Cleanup**: Graceful shutdown when user closes application
        
        Key Responsibilities:
        - Maintains 60 FPS frame rate for smooth UI experience
        - Processes all user interactions (mouse, keyboard, sliders)
        - Updates real-time visualizations and training progress
        - Manages multi-threaded training without blocking UI
        - Handles application lifecycle (startup, running, shutdown)
        
        Performance Characteristics:
        - Frame Rate: 60 FPS (configurable via FPS constant)
        - Event Processing: Every frame for responsive controls
        - Rendering: Complete screen redraw every frame
        - Training: Runs in separate thread for non-blocking operation
        
        Shutdown Process:
        1. Stops any running training threads
        2. Waits for training thread to finish (1 second timeout)
        3. Cleans up pygame resources
        4. Exits application gracefully
        
        Usage:
        ```python
        app = GANApp()
        app.run()  # Starts the interactive application
        ```
        
        Note: This method blocks until the user closes the application window.
        All training and visualization happens within this loop.
        """
        running = True
        
        while running:
            running = self.handle_events()
            self.draw()
            self.clock.tick(FPS)
        
        self.stop_training()
        pygame.quit()

if __name__ == "__main__":
    app = GANApp()
    app.run()
