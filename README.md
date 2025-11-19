
Synthetic Microstructure Generation using WGAN-GP
Training Animation

A Wasserstein Generative Adversarial Network with Gradient Penalty (WGAN-GP) implementation for synthesizing high-fidelity Voronoi material microstructures using TensorFlow/Keras. This project leverages advanced optimization techniques—including Mixed Precision training and Two-Time-Scale Update Rules—to address "mode collapse" and generate scientifically accurate grain boundaries for computational material science.

🎯 Objective
Design and implement a WGAN-GP to synthesize sharp, diverse Voronoi tessellations mimicking microscopic material microstructures. By replacing traditional GAN losses with Wasserstein distance and gradient penalties, the model achieves stable training and superior sample quality over DCGAN baselines.

1. 🏗️ Architecture & Model Design
Approach
Generator: Deep upsampling network (Latent Dim → 64×64) utilizing UpSampling2D and Conv2D. Deliberately excludes Batch Normalization to prevent artifacting in texture generation.
Critic (Discriminator): Deep convolutional network with strided convolutions, LeakyReLU activations, and linear output (no Sigmoid) to approximate the Wasserstein distance.
WGAN-GP Loss: Implements Wasserstein loss with Gradient Penalty (λ=10) to enforce the 1-Lipschitz constraint for stable gradients.
Weight Initialization: Random Normal initialization aligned with WGAN convergence requirements.
Key Achievements
Solved "mode collapse" issues inherent in DCGANs, producing diverse grain structures.
Achieved stable adversarial training where Critic loss correlates with generation quality.
Generated sharp, realistic grain boundaries and triple junctions without blurring.
Model Summary (Generator Parameters: ~4.2M | Critic: ~2.8M):

Component	Input Shape	Output Shape	Key Layers
Generator	(100,)	(64, 64, 1)	Dense(8192) → UpSample(4→8) → Conv(256) → ... → Tanh
Critic	(64, 64, 1)	(1,)	Conv(64, stride=2) → LeakyReLU → ... → Dense(1)
2. 🔄 Data Pipeline & Preprocessing
Approach
Data Ingestion: Automated pipeline handling H5 file parsing and stacking from compressed Voronoi datasets.
Normalization Strategy: Dynamic normalization mapping pixel values to [-1, 1] to align with the Generator's tanh activation output.
TF Data Pipeline: Optimized tf.data.Dataset with shuffling and batching for efficient GPU streaming.
Key Achievements
Seamless handling of high-dimensional microstructure datasets.
Optimized memory usage allowing for larger batch sizes (64) on consumer GPUs.
Robust preprocessing ensuring numerical stability during high-variance GAN training.
Sample Dataset Visualization:
Dataset Samples
(9 random 64×64 Voronoi images from the loaded dataset.)

3. ⚙️ Training Infrastructure & Monitoring
Problem Solved
GANs are notoriously unstable. This project implements a custom training loop with specialized scheduling to ensure the Critic stays ahead of the Generator.

Approach
Two-Time-Scale Update Rule (TTUR): Implements distinct learning rates for Generator (1e⁻⁴) and Critic (4e⁻⁴).
Cosine Decay Scheduler: Adaptive learning rate scheduling for both networks to fine-tune convergence in later epochs.
Automated Checkpointing: System to save/restore model states, allowing training resumption after interruptions.
Visual Monitoring: Automatic generation of GIF frames every epoch to track the "coarse-to-fine" learning process.
Key Achievements
Robust training loop with automatic crash recovery/resume.
Clear visual documentation of microstructural evolution via animated GIFs.
Stable loss convergence (~ -25) indicating continuous learning.
Training Evolution
(Generated microstructures from Epoch 0 to 100; noise-to-structure transformation.)

4. 🔧 Technical Implementation & Optimization
Programming Language
Python 3.8+

Deep Learning Stack
TensorFlow 2.x / Keras: Core framework for graph construction.
Mixed Precision (AMP): Enabled mixed_float16 policy to reduce VRAM usage by ~50% and accelerate training throughput on NVIDIA GPUs.
NumPy & h5py: High-performance numerical computation and data handling.
ImageIO: For generating training progression animations.
Advanced Features
Gradient Penalty: Manually computed gradients of the Critic with respect to interpolated images to enforce Lipschitz continuity.
Loss Scaling: Integrated LossScaleOptimizer to prevent underflow during mixed-precision training.
Custom Train Step: Overrode standard train_step to implement specific WGAN-GP logic.
Hyperparameters Table:

Parameter	Value	Rationale
Batch Size	64	Balances stability and efficiency
Epochs	100	Sufficient for convergence
Latent Dim	100	Rich noise for diversity
β₁/β₂ (Adam)	0.5 / 0.9	Momentum for GANs
GP Weight (λ)	10	Standard for WGAN-GP
5. 📊 Key Results & Performance
Sample Output: Synthesized 64×64 microstructures indistinguishable from ground truth Voronoi tessellations.
Training Stability: Critic loss converged stably without oscillation, providing a reliable metric for optimization.
Efficiency: High-performance pipeline capable of training on T4 GPUs within reasonable timeframes due to AMP optimizations.
Impact
Enables rapid generation of synthetic material datasets for downstream ML models.
Provides a foundation for Conditional GANs (cGAN) for property-driven material design.
Demonstrates mastery of advanced Generative AI optimization techniques.
Loss Convergence Plot (Conceptual):
Loss Curves
(Critic loss (blue) stabilizes; Generator (orange) minimizes steadily.)

6. 🚀 Technical Achievements
Generative AI: Implementation of WGAN-GP, one of the most stable GAN architectures.
Model Optimization: Usage of Mixed Precision (AMP) and Learning Rate Schedulers.
Production Features: Checkpointing, automated GIF generation, and modular code structure.
Visualization: Real-time tracking of grain boundary formation.
📖 Usage
Prerequisites
Google Colab (T4 GPU recommended) or Local CUDA Environment
Dataset: Voronoi_micro_imgs.zip (Uploaded to root)
Setup
Install Dependencies:
Bash
!pip install imageio tensorflow-docs -q
Run the Training Script:
Python
# Execute cells sequentially.
# The script handles zip extraction, data loading, and model training.
Resume Training:
Python
# Checkpoints are saved to ./checkpoints
# Re-running the script automatically detects and restores the latest checkpoint.
Generated Output
Checkpoints: Saved locally in ./checkpoints.
Visuals: Progress frames saved to ./gif_frames.
Animation: Final training evolution saved as voronoi_training_animation.gif.
Metrics: Loss curves for Generator and Critic.
Customization
Adjust epochs or batch_size in the config dictionary.
Modify gp_weight (Gradient Penalty weight) to tune training stability.
Change z_noise_dim to alter the latent space complexity.
