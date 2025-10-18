# Interactive Fashion GAN

## Real-Time DCGAN Training with Pygame Interface

An interactive PyTorch-based Generative Adversarial Network (GAN) application for generating 28×28 fashion product images, featuring a real-time pygame interface with comprehensive controls for training and parameter adjustment.

## 🎯 Project Overview

This project implements a **Deep Convolutional GAN (DCGAN)** specifically designed for fashion image generation. It combines the power of PyTorch's deep learning capabilities with an intuitive pygame interface, allowing users to interactively train, visualize, and experiment with GAN models in real-time.

### Key Features

- **🔄 Real-time Training**: Watch the GAN train with live loss curves and image generation
- **🎮 Interactive Controls**: Start/stop training, generate images, save/load models
- **⚙️ Dynamic Parameter Adjustment**: Modify all training parameters during runtime
- **🖼️ Live Image Generation**: See generated images update as training progresses
- **🎲 Noise Seed Control**: Reproducible image generation with adjustable noise seeds
- **💾 Model Persistence**: Save and load trained models with checkpoints
- **📊 Comprehensive Monitoring**: Real-time progress tracking and statistics

## 🏗️ Architecture & Algorithms

### DCGAN Architecture

The application uses a **Deep Convolutional GAN (DCGAN)** architecture optimized for 28×28 grayscale images:

#### Generator Network

```
Input: 100D noise vector (z_dim configurable: 50-200)
├── Fully Connected Layer: z_dim → base*7*7
├── Batch Normalization + ReLU
├── Reshape: [batch_size, base, 7, 7]
├── ConvTranspose2d: 7×7 → 14×14 (stride=2)
├── Batch Normalization + ReLU
├── ConvTranspose2d: 14×14 → 28×28 (stride=2)
└── Tanh Activation: Output [-1, 1]
```

#### Discriminator Network

```
Input: 28×28 grayscale images
├── Conv2d: 28×28 → 14×14 (stride=2)
├── LeakyReLU(0.2)
├── Conv2d: 14×14 → 7×7 (stride=2)
├── Batch Normalization + LeakyReLU(0.2)
├── Flatten: [batch_size, base*2*7*7]
└── Linear: → 1 (binary classification logit)
```

### Training Algorithm

The training process implements the **adversarial training paradigm**:

1. **Discriminator Training**:

   - Real images: Trained to output high scores (label = 1)
   - Fake images: Trained to output low scores (label = 0)
   - Loss: Binary Cross-Entropy with Logits

2. **Generator Training**:

   - Fake images: Trained to fool discriminator (label = 1)
   - Loss: Binary Cross-Entropy with Logits (inverted)

3. **Optimization**:
   - Adam optimizer with β1=0.5, β2=0.999
   - Learning rate: 0.0002 (adjustable via slider)
   - Batch size: 64 (adjustable via slider)

## 📦 Installation & Requirements

### Prerequisites

- Python 3.7+
- PyTorch 1.8+
- Pygame 2.0+

### Installation

1. **Clone the repository**:

```bash
git clone <repository-url>
cd UrbanStyle/fashion-gan
```

2. **Install dependencies**:

```bash
pip install -r requirements.txt
```

3. **Run the application**:

```bash
python gan_app.py
```

### Dependencies

```
torch>=1.8.0
torchvision>=0.9.0
pygame>=2.0.0
Pillow>=8.0.0
numpy>=1.19.0
matplotlib>=3.3.0
```

## 🎮 User Interface Guide

### Control Buttons

| Button             | Function               | Description                                    |
| ------------------ | ---------------------- | ---------------------------------------------- |
| **Start Training** | `on_start_training()`  | Begins GAN training process in separate thread |
| **Stop Training**  | `on_stop_training()`   | Gracefully halts current training              |
| **Generate**       | `on_generate_images()` | Generates new images with current noise seed   |
| **Save Model**     | `on_save_model()`      | Saves current model state to checkpoints/      |
| **Load Model**     | `on_load_model()`      | Loads previously saved model                   |

### Parameter Sliders (Real-Time Updates)

| Slider              | Range         | Default | Effect                           | Impact                       |
| ------------------- | ------------- | ------- | -------------------------------- | ---------------------------- |
| **Learning Rate**   | 0.0001 - 0.01 | 0.0002  | Updates optimizer learning rates | Training speed & convergence |
| **Noise Seed**      | 1 - 100       | 42      | Controls reproducible generation | Different image styles       |
| **Training Epochs** | 1 - 10        | 1       | Sets maximum training epochs     | Training duration            |
| **Batch Size**      | 32 - 256      | 64      | Recreates data loader            | Memory usage & dynamics      |
| **Noise Dimension** | 50 - 200      | 100     | Recreates models                 | Model capacity & diversity   |

### Visualizations

- **🖼️ Generated Images Grid**: 4×4 real-time display of generated fashion images
- **📈 Loss Curves**: Live plotting of Generator and Discriminator losses
- **📊 Training Status**: Current epoch, batch count, and progress indicators
- **🎯 Parameter Display**: Real-time parameter values and training statistics

## 🔧 Technical Implementation

### Multi-Threading Architecture

The application uses a **multi-threaded design** to maintain UI responsiveness:

- **Main Thread**: Handles pygame events, UI updates, and parameter changes
- **Training Thread**: Runs GAN training loop without blocking UI
- **Thread Communication**: Uses shared state variables for coordination

### Real-Time Parameter Updates

All slider changes are applied **immediately** during training:

```python
# Runs every frame (60 FPS)
def handle_events(self):
    slider_values = self.slider_panel.get_values()

    # Learning rate changes optimizer immediately
    if new_lr != self.lr:
        self.optimizer_G = optim.Adam(..., lr=new_lr)
        self.optimizer_D = optim.Adam(..., lr=new_lr)

    # Batch size recreates data loader
    if new_batch_size != self.batch_size:
        self.setup_data()  # Recreates with new batch size

    # Noise dimension recreates models
    if new_z_dim != self.z_dim:
        self.generator = Generator(new_z_dim)
        self.discriminator = Discriminator()
```

### Batch-Based Epoch Tracking

The training system uses **batch-based epoch tracking** for accurate progress display:

```python
# Calculate epoch based on batches processed
current_epoch = (batch_count // self.batches_per_epoch) + 1
total_batches_needed = self.max_epochs * self.batches_per_epoch
```

## 📁 Project Structure

```
UrbanStyle/fashion-gan/
├── gan_app.py              # Main interactive application
├── train_gan.py            # Original command-line training script
├── requirements.txt        # Python dependencies
├── README.md              # This documentation
├── data/                  # Custom image data (optional)
│   └── custom_images/     # Place custom training images here
├── FashionMNIST/          # Fashion-MNIST dataset (auto-downloaded)
│   ├── raw/              # Raw dataset files
│   └── processed/        # Processed dataset files
├── checkpoints/           # Saved model states
│   ├── generator_epoch_X.pth
│   ├── discriminator_epoch_X.pth
│   └── samples_epoch_X.png
└── out/                  # Training outputs and visualizations
    ├── loss_curves.png
    └── generated_samples/
```

## 🎯 Usage Examples

### Basic Training Workflow

1. **Start the application**:

```bash
python gan_app.py
```

2. **Adjust parameters** (optional):

   - Set Training Epochs to 5
   - Adjust Learning Rate to 0.001
   - Change Batch Size to 128

3. **Begin training**:

   - Click "Start Training"
   - Watch real-time progress and generated images

4. **Generate images**:

   - Adjust Noise Seed slider
   - Click "Generate" to see different styles

5. **Save progress**:
   - Click "Save Model" to preserve current state

### Advanced Parameter Tuning

| Scenario               | Recommended Settings         | Reasoning                             |
| ---------------------- | ---------------------------- | ------------------------------------- |
| **Fast Training**      | LR: 0.001, Batch: 128        | Higher learning rate + larger batches |
| **High Quality**       | LR: 0.0001, Epochs: 10       | Lower LR for stability, more epochs   |
| **Memory Limited**     | Batch: 32, Noise Dim: 50     | Smaller batches and model size        |
| **Diverse Generation** | Noise Dim: 200, Seed: Random | Larger latent space for diversity     |

## 🔍 Monitoring & Debugging

### Training Indicators

- **Generator Loss**: Should decrease over time (better at fooling discriminator)
- **Discriminator Loss**: Should decrease over time (better at detecting fakes)
- **Generated Images**: Should become more realistic and diverse
- **Epoch Progress**: Should increment smoothly through training

### Common Issues & Solutions

| Issue                  | Symptoms                          | Solution                             |
| ---------------------- | --------------------------------- | ------------------------------------ |
| **Training Too Slow**  | Loss decreases very slowly        | Increase learning rate or batch size |
| **Training Unstable**  | Loss oscillates wildly            | Decrease learning rate               |
| **Poor Image Quality** | Generated images are blurry/noisy | Increase training epochs             |
| **Memory Errors**      | Application crashes               | Reduce batch size or noise dimension |
| **CUDA Issues**        | GPU not detected                  | App automatically falls back to CPU  |

## 🚀 Performance Optimization

### Hardware Recommendations

- **GPU**: NVIDIA GPU with 4GB+ VRAM (CUDA support)
- **RAM**: 8GB+ system memory
- **CPU**: Multi-core processor for data loading

### Performance Tips

1. **Use GPU**: Ensure CUDA is available for faster training
2. **Batch Size**: Larger batches = faster training but more memory
3. **Data Loading**: Increase `num_workers` for faster data loading
4. **Model Size**: Adjust `base` parameter for Generator/Discriminator capacity

## 📚 Algorithm Details

### Loss Functions

**Discriminator Loss**:

```
L_D = -E[log(D(x_real))] - E[log(1 - D(G(z)))]
```

**Generator Loss**:

```
L_G = -E[log(D(G(z)))]
```

### Weight Initialization

- **Convolutional Layers**: Normal distribution (μ=0, σ=0.02)
- **Batch Normalization**: Normal distribution (μ=1, σ=0.02)
- **Bias Terms**: Initialized to zero

### Data Preprocessing

- **Normalization**: Pixel values scaled from [0, 1] to [-1, 1]
- **Augmentation**: Resize and center crop to 28×28
- **Batch Processing**: Shuffled batches with drop_last=True

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Fashion-MNIST Dataset**: Zalando Research
- **DCGAN Paper**: "Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks"
- **PyTorch Community**: For the excellent deep learning framework
- **Pygame Community**: For the game development library

---

**Happy Training! 🎨✨**
