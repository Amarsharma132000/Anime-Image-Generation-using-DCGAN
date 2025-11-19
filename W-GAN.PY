CELL 1 ( # FOR DATA LOADING):
!unzip /content/Voronoi_micro_imgs-20251103T050538Z-1-001.zip -d /content/unzipped_data
 
import h5py
import os
import numpy as np

unzipped_dir = '/content/unzipped_data/Voronoi_micro_imgs'
file_list = [os.path.join(unzipped_dir, f) for f in os.listdir(unzipped_dir) if f.endswith('.h5')]
 
with h5py.File(file_list[0], 'r') as f:
    keys = list(f.keys())
    data_shape = f[keys[0]].shape
    dtype = f[keys[0]].dtype
 
all_data = np.zeros((len(file_list),) + data_shape, dtype=dtype)
 
for i, file_path in enumerate(file_list):
    with h5py.File(file_path, 'r') as f:
        all_data[i, ...] = f[keys[0]][:]
    if (i + 1) % 1000 == 0:
        print(f'{i+1} files read')
 
print("All data loaded successfully.")
print(f"Shape of combined data: {all_data.shape}")
 
CELL 2 (# FOR VISUALIZING THE DATASET):
import matplotlib.pyplot as plt
import random
 
fig, axs = plt.subplots(3, 3, figsize=(10, 10))
num_images_to_plot = 9
indices_to_plot = random.sample(range(all_data.shape[0]), num_images_to_plot)
 
for i, ax in enumerate(axs.flatten()):
    image_data = all_data[indices_to_plot[i], :, :, 0] # Assuming the image is 64x64 and the channel is the last dim
 
    ax.imshow(image_data, cmap='gray') # Assuming grayscale images
    ax.set_title(f"Image {indices_to_plot[i]}")
    ax.axis('off')
 
plt.tight_layout()
plt.show()
 
CELL 3 (#  TRAINING ):
!pip install imageio -q
!pip install tensorflow-docs -q
 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import os
import h5py
import PIL.Image
import imageio.v2 as imageio      # Use v2 to avoid deprecation warnings
import glob
import matplotlib.pyplot as plt
from IPython.display import display
import tensorflow_docs.vis.embed as embed
 
# --- 1. SETUP & CONFIGURATION ---
 
# --- A. ENABLE AUTOMATIC MIXED PRECISION (AMP) ---
print("Enabling Mixed Precision (AMP)...")
tf.keras.mixed_precision.set_global_policy('mixed_float16')
 
# --- B. CONFIGURATION ---
config = {
    "epochs": 100,
    "batch_size": 64,
 
    # --- SOTA FIX (TTUR + Decay): Different LRs for G and D ---
    "g_learning_rate": 0.0001, # Generator Initial LR
    "d_learning_rate": 0.0004, # Critic Initial LR (4x faster)
    # --------------------------------------------------------
   
    "beta_1": 0.5,
    "beta_2": 0.9,
    "z_noise_dim": 100,
    "image_size": 64,
    "channels": 1,
    "checkpoint_interval": 10,
    "model_type": "WGAN",
   
    # (TTUR + Decay): (n_critic is 1)
    "n_critic": 1,
    "gp_weight": 10.0
}
 
# --- C. LOCAL DIRECTORIES ---
GIF_FRAME_DIR = './gif_frames'
os.makedirs(GIF_FRAME_DIR, exist_ok=True)
 
 
# --- 2. DATA PREPARATION ---
print("Unzipping and loading data...")
# --- NOTE: Ensure the zip file is uploaded to the Colab root directory ---
!unzip -o /content/Voronoi_micro_imgs-20251103T050538Z-1-001.zip -d /content/unzipped_data
 
print("Loading and reshaping H5 files...")
unzipped_dir = '/content/unzipped_data/Voronoi_micro_imgs'
file_list = [os.path.join(unzipped_dir, f) for f in os.listdir(unzipped_dir) if f.endswith('.h5')]
 
all_data_list = []
data_key = None
for file_path in file_list:
    with h5py.File(file_path, 'r') as f:
        if data_key is None:
            data_key = list(f.keys())[0]
       
all_data_list.append(f[data_key][:])
 
stacked_data = np.stack(all_data_list, axis=0)
print(f"Initial stacked shape: {stacked_data.shape}")
 
x_train = np.transpose(stacked_data, (0, 3, 1, 2))
x_train = x_train.reshape(-1, 64, 64)
x_train = np.expand_dims(x_train, axis=-1)
 
print(f"Final COMBINED data shape (N, H, W, C): {x_train.shape}")
 
print("\n--- Data Range Check ---")
print(f"Data type: {x_train.dtype}")
print(f"Min value in data: {np.min(x_train)}")
print(f"Max value in data: {np.max(x_train)}")
print("--------------------------\n")
 
if np.max(x_train) <= 1.0:
    print("Normalizing for [0, 1] range.")
    x_train_normalized = (x_train.astype('float32') - 0.5) / 0.5
else:
    print("Normalizing for [0, 255] range.")
    x_train_normalized = (x_train.astype('float32') - 127.5) / 127.5
 
train_dataset = tf.data.Dataset.from_tensor_slices(x_train_normalized).shuffle(x_train.shape[0]).batch(config['batch_size'], drop_remainder=True)
 
# --- 3. MODEL ARCHITECTURE (STABLE) ---
# Using UpSampling2D + Conv2D and NO BatchNormalization in Generator
def build_generator():
    noise_input = layers.Input(shape=(config['z_noise_dim'],))
    x = layers.Dense(4 * 4 * 512)(noise_input)
    x = layers.Reshape((4, 4, 512))(x)
 
    # 8x8
    x = layers.UpSampling2D()(x)
    x = layers.Conv2D(256, kernel_size=3, strides=1, padding='same')(x)
    x = layers.LeakyReLU(negative_slope=0.2)(x)
 
    # 16x16
    x = layers.UpSampling2D()(x)
    x = layers.Conv2D(128, kernel_size=3, strides=1, padding='same')(x)
    x = layers.LeakyReLU(negative_slope=0.2)(x)
 
    # 32x32
    x = layers.UpSampling2D()(x)
    x = layers.Conv2D(64, kernel_size=3, strides=1, padding='same')(x)
    x = layers.LeakyReLU(negative_slope=0.2)(x)
   
    # 64x64
    x = layers.UpSampling2D()(x)
    output_image = layers.Conv2D(config['channels'], kernel_size=3, strides=1, padding='same')(x)
   
    output_image = layers.Activation('tanh', dtype='float32')(output_image)
 
    return keras.Model(noise_input, output_image)
 
def build_critic():
    image_input = layers.Input(shape=(config['image_size'], config['image_size'], config['channels']))
 
    x = layers.Conv2D(64, kernel_size=4, strides=2, padding='same')(image_input) # 32x32
    x = layers.LeakyReLU(negative_slope=0.2)(x)
 
    x = layers.Conv2D(128, kernel_size=4, strides=2, padding='same')(x) # 16x16
    x = layers.LeakyReLU(negative_slope=0.2)(x)
 
    x = layers.Conv2D(256, kernel_size=4, strides=2, padding='same')(x) # 8x8
    x = layers.LeakyReLU(negative_slope=0.2)(x)
 
    x = layers.Conv2D(512, kernel_size=4, strides=2, padding='same')(x) # 4x4
    x = layers.LeakyReLU(negative_slope=0.2)(x)
 
    x = layers.Flatten()(x)
    output = layers.Dense(1)(x)
    output = layers.Activation('linear', dtype='float32')(output)
 
    return keras.Model(image_input, output)
 
# --- 4. MODEL & OPTIMIZER SETUP (TTUR + Decay) ---
print("Setting up optimizers with TTUR + CosineDecay Schedulers...")
 
# ---Create two decaying LR schedules ---
try:
    steps_per_epoch = x_train.shape[0] // config['batch_size']
    total_steps = steps_per_epoch * config['epochs']
    print(f"Total training steps for LR schedule: {total_steps}")
except Exception as e:
    total_steps = 31200 # Fallback (20000 / 64 * 100)
    print(f"Warning: Could not calculate steps. Defaulting to {total_steps} steps.")
 
g_lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=config['g_learning_rate'],
    decay_steps=total_steps,
    alpha=0.0
)
 
d_lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=config['d_learning_rate'],
    decay_steps=total_steps,
    alpha=0.0
)
# ----------------------------------------------------
 
critic = build_critic()
generator = build_generator()
 
# --- Both optimizers use their *own* decaying schedule ---
g_optimizer_base = tf.keras.optimizers.Adam(learning_rate=g_lr_schedule, beta_1=config['beta_1'], beta_2=config['beta_2'])
d_optimizer_base = tf.keras.optimizers.Adam(learning_rate=d_lr_schedule, beta_1=config['beta_1'], beta_2=config['beta_2'])
# ----------------------------------------------------------------------
 
g_optimizer = tf.keras.mixed_precision.LossScaleOptimizer(g_optimizer_base)
d_optimizer = tf.keras.mixed_precision.LossScaleOptimizer(d_optimizer_base)
 
# --- 5. CHECKPOINTING & RESUME SETUP ---
start_epoch = tf.Variable(0, dtype=tf.int64)
checkpoint_dir = './checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)
checkpoint_prefix = os.path.join(checkpoint_dir, "model_checkpoint")
 
ckpt = tf.train.Checkpoint(generator=generator,
                           critic=critic,
                           g_optimizer=g_optimizer_base,
                           d_optimizer=d_optimizer_base,
                           epoch=start_epoch)
 
# --- Check for local checkpoint ---
latest_ckpt = tf.train.latest_checkpoint(checkpoint_dir)
if latest_ckpt:
    try:
        ckpt.restore(latest_ckpt)
        print(f"Local checkpoint restored from epoch {start_epoch.numpy()}")
    except Exception as e:
        print(f"Could not restore checkpoint. Starting from scratch. Error: {e}")
else:
    print("No local checkpoint found. Starting from scratch.")
 
 
# --- 6. LOSS FUNCTIONS & TRAINING STEPS ---
def critic_loss_wgan(real_output, fake_output):
    return tf.reduce_mean(fake_output) - tf.reduce_mean(real_output)
 
def generator_loss_wgan(fake_output):
    return -tf.reduce_mean(fake_output)
 
def gradient_penalty(real_images, fake_images):
    current_batch_size = tf.shape(real_images)[0]
    alpha = tf.random.uniform([current_batch_size, 1, 1, 1], 0.0, 1.0)
    alpha = tf.cast(alpha, real_images.dtype)
    interpolated = real_images + alpha * (fake_images - real_images)
 
    with tf.GradientTape() as gp_tape:
        gp_tape.watch(interpolated)
        pred = critic(interpolated, training=True)
 
    grads = gp_tape.gradient(pred, [interpolated])[0]
    norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
    gp = tf.reduce_mean((norm - 1.0) ** 2)
    return gp
 
@tf.function
def train_step_wgan(images):
    current_batch_size = tf.shape(images)[0]
    noise = tf.random.normal([current_batch_size, config['z_noise_dim']])
 
    # --- Train Critic ---
    # The loop will run once as config['n_critic'] = 1
    for _ in range(config['n_critic']):
        with tf.GradientTape() as tape:
            fake_images = generator(noise, training=True)
            real_output = critic(images, training=True)
            fake_output = critic(fake_images, training=True)
            c_loss = critic_loss_wgan(real_output, fake_output)
            gp = gradient_penalty(images, fake_images)
           
            total_critic_loss = c_loss + config['gp_weight'] * gp
 
        c_grads = tape.gradient(total_critic_loss, critic.trainable_variables)
        d_optimizer.apply_gradients(zip(c_grads, critic.trainable_variables))
 
    # --- Train Generator ---
    with tf.GradientTape() as tape:
        fake_images = generator(noise, training=True)
        fake_output = critic(fake_images, training=True)
        gen_loss = generator_loss_wgan(fake_output)
 
    g_grads = tape.gradient(gen_loss, generator.trainable_variables)
    g_optimizer.apply_gradients(zip(g_grads, generator.trainable_variables))
 
    return gen_loss, total_critic_loss
 
# --- 7. HELPER FUNCTIONS FOR VISUALIZATION ---
 
# --- HELPER FOR GIF (16 IMAGES) ---
num_examples_to_generate = 16
seed = tf.random.normal([num_examples_to_generate, config['z_noise_dim']])
 
def generate_and_save_images(model, epoch, test_input):
    predictions = model(test_input, training=False)
    predictions_denorm = (predictions * 127.5 + 127.5).numpy().astype(np.uint8)
 
    fig = plt.figure(figsize=(8, 8))
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions_denorm[i, :, :, 0], cmap='gray')
        plt.axis('off')
 
    frame_filepath = os.path.join(GIF_FRAME_DIR, f'image_at_epoch_{epoch:04d}.png')
    plt.savefig(frame_filepath)
    plt.close(fig)
    # Removed wandb.log()
 
# --- 8. TRAINING LOOP ---
print(f"Starting/Resuming training from epoch {start_epoch.numpy()}")
 
for epoch in range(start_epoch.numpy(), config['epochs']):
    start_epoch.assign_add(1)
   
    g_loss_epoch = 0.0
    d_loss_epoch = 0.0
    num_batches = 0
 
    for image_batch in train_dataset:
        g_loss, d_loss = train_step_wgan(image_batch)
        g_loss_epoch += g_loss.numpy()
        d_loss_epoch += d_loss.numpy()
        num_batches += 1
 
    # --- Log both decaying learning rates ---
    current_g_lr = g_lr_schedule(g_optimizer_base.iterations).numpy()
    current_d_lr = d_lr_schedule(d_optimizer_base.iterations).numpy()
 
    # Checkpoint and save GIF frame
    if (epoch + 1) % config['checkpoint_interval'] == 0:
        ckpt.save(file_prefix=checkpoint_prefix)
 
        avg_g_loss = g_loss_epoch / num_batches
        avg_d_loss = d_loss_epoch / num_batches
 
        print(f'\nEpoch {epoch + 1}, G Loss={avg_g_loss}, D Loss={avg_d_loss}')
        print(f'Current G_LR: {current_g_lr} | Current D_LR: {current_d_lr}')
        print(f"Checkpoint saved locally for epoch {epoch+1}")
 
        generate_and_save_images(generator, epoch + 1, seed)
        print(f"Saved GIF frame for epoch {epoch + 1}")
 
# --- FINAL GENERATION ---
generate_and_save_images(generator, config['epochs'], seed)
 
print("Training finished.")
 
# --- 9. CREATE AND DISPLAY THE GIF ---
print("\nCreating training GIF...")
anim_file = 'voronoi_training_animation.gif'
 
with imageio.get_writer(anim_file, mode='I') as writer:
    filenames = sorted(glob.glob(os.path.join(GIF_FRAME_DIR, 'image*.png')))
    print(f"Found {len(filenames)} frames for GIF.")
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)
    if filenames:
        image = imageio.imread(filenames[-1]) # Add last frame again for pause
        writer.append_data(image)
 
print(f"GIF saved to {anim_file}")
 
# Embed the GIF in the Colab notebook
display(embed.embed_file(anim_file))
