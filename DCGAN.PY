import os
import json
import numpy as np
import matplotlib.pyplot as plt
import warnings
import tqdm.notebook as tqdm

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img,array_to_img
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping

warnings.filterwarnings("ignore")

# Load Kaggle credentials from Colab secrets
from google.colab import userdata
from google.colab.userdata import SecretNotFoundError

try:
    kaggle_credentials = json.loads(userdata.get('KAGGLE_KEY'))
    os.environ['KAGGLE_USERNAME'] = kaggle_credentials['username']
    os.environ['KAGGLE_KEY'] = kaggle_credentials['key']
    print("Kaggle credentials loaded successfully.")
except SecretNotFoundError:
    print("Kaggle API key not found in Colab secrets. Please add it using the instructions provided.")
    # Exit or handle the error appropriately
    exit()
except KeyError as e:
    print(f"Error accessing Kaggle credential: {e}. Please check your KAGGLE_KEY secret format.")
    # Exit or handle the error appropriately
    exit()

# Check if environment variables are set
if 'KAGGLE_USERNAME' not in os.environ or 'KAGGLE_KEY' not in os.environ:
    print("Kaggle environment variables are not set. Please ensure your KAGGLE_KEY secret is correctly configured in Colab Secrets and notebook access is enabled.")
    exit()


# Download and unzip the data
!kaggle datasets download 'soumikrakshit/anime-faces'
!unzip /content/anime-faces.zip -d /content/

base_dir= "/content/data/data"

# loading image path list
image_paths=[]
for image_name in os.listdir(base_dir):
  image_path=os.path.join(base_dir,image_name)
  image_paths.append(image_path)

# Reduce the dataset size for faster training (e.g., use only the first 5000 images)
image_paths = image_paths[:3000]

# load the image and convert to numpy array
def load_image(path):
  image = tf.io.read_file(path)
  image = tf.image.decode_jpeg(image, channels=3)  # Adjust channels if needed
  image = tf.image.convert_image_dtype(image, dtype=tf.float32)
  return image

train_dataset = tf.data.Dataset.from_tensor_slices(image_paths)
train_dataset = train_dataset.map(load_image)  # Apply load_image function to each path
train_dataset = train_dataset.batch(32)  # Adjust batch size as needed

# Convert to NumPy array
train_images = []
for batch in train_dataset:
    for image in batch:
        train_images.append(image.numpy())

train_images = np.array(train_images)

# Normalize the pixel values to the range [-1, 1]
train_images = (train_images - 127.5) / 127.5

# latent dimension for random noise
latent_dim=100

# weight initialiser
weight_init=keras.initializers.RandomNormal(mean=0.0,stddev=0.02)

# no of channels
channels=3

# Generator
model=Sequential(name="Generator")
#  1d random noise
model.add(layers.Dense(8*8*512,use_bias=False,input_dim=latent_dim))
model.add(layers.BatchNormalization())
model.add(layers.ReLU())

# convert 1D to 3D
model.add(layers.Reshape((8,8,512)))

# upsample to 16x16
model.add(layers.Conv2DTranspose(256,kernel_size=(4,4),strides=(2,2),padding="same",kernel_initializer=weight_init))
model.add(layers.BatchNormalization())
model.add(layers.ReLU())

# upsample to 32x32
model.add(layers.Conv2DTranspose(128,kernel_size=(4,4),strides=(2,2),padding="same",kernel_initializer=weight_init))
model.add(layers.BatchNormalization())
model.add(layers.ReLU())

# upsample to 64x64
model.add(layers.Conv2DTranspose(64,kernel_size=(4,4),strides=(2,2),padding="same",kernel_initializer=weight_init))
model.add(layers.BatchNormalization())
model.add(layers.ReLU())

model.add(layers.Conv2D(channels,kernel_size=(4,4),padding="same",activation="tanh"))

generator=model

# Discriminator model
model=Sequential(name="Discriminator")
input_shape=(64,64,3)
alpha=0.2

# create conv layers
model.add(layers.Conv2D(64,kernel_size=(4,4),strides=(2,2),padding="same",input_shape=input_shape))
model.add(layers.BatchNormalization())
model.add(layers.LeakyReLU(alpha=alpha))

model.add(layers.Conv2D(128,kernel_size=(4,4),strides=(2,2),padding="same",input_shape=input_shape))
model.add(layers.BatchNormalization())
model.add(layers.LeakyReLU(alpha=alpha))

model.add(layers.Conv2D(128,kernel_size=(4,4),strides=(2,2),padding="same",input_shape=input_shape))
model.add(layers.BatchNormalization())
model.add(layers.LeakyReLU(alpha=alpha))

model.add(layers.Flatten())
model.add(layers.Dropout(0.3))

# output class
model.add(layers.Dense(1,activation="sigmoid"))

discriminator=model

# Create DCGAN
class DCGAN(keras.Model):
  def __init__(self,discriminator,generator,latent_dim):
    super().__init__()
    self.discriminator=discriminator
    self.generator=generator
    self.latent_dim=latent_dim
    self.g_loss_metric=keras.metrics.Mean(name="g_loss")
    self.d_loss_metric=keras.metrics.Mean(name="d_loss")

  @property
  def metrics(self):
    return [self.g_loss_metric,self.d_loss_metric]

  def compile(self,d_optimizer,g_optimizer,loss_fn):
    super().compile()
    self.d_optimizer=d_optimizer
    self.g_optimizer=g_optimizer
    self.loss_fn=loss_fn

  def build(self, input_shape):
      # Build the generator with a dummy input
      self.generator.build((input_shape[0], self.latent_dim))
      # Build the discriminator with a dummy input
      self.discriminator.build(input_shape)
      super().build(input_shape)


  def train_step(self,real_images):
    # get batchsize from the data
    batch_size=tf.shape(real_images)[0]

    # create random noise
    random_noise=tf.random.normal(shape=(batch_size,self.latent_dim))

    # train the discriminator with real(1) and fake(0) images
    with tf.GradientTape() as tape:
      # compute loss on real images
      pred_real=self.discriminator(real_images,training=True)
      # generate real image labels
      real_labels=tf.ones((batch_size,1))
      # one sided label smoothing
      real_labels+=0.05*tf.random.uniform(tf.shape(real_labels)) # will generate values between [0-1] with shape of real labels
      # we multiply by 0.05 to apply label smoothing

      # compute loss
      d_loss_real=self.loss_fn(real_labels,pred_real)

      # compute loss on fake images
      fake_images=self.generator(random_noise)
      pred_fake=self.discriminator(fake_images,training=True)

      # generate fake image labels
      fake_labels=tf.zeros((batch_size,1))

      # compute loss
      d_loss_fake=self.loss_fn(fake_labels,pred_fake)

      # total discriminator loss
      d_loss=(d_loss_real+d_loss_fake)/2

    # compute discriminator gradients
    gradients=tape.gradient(d_loss,self.discriminator.trainable_variables)
    # update the gradients
    self.d_optimizer.apply_gradients(zip(gradients,self.discriminator.trainable_variables))

    # train the  GENERATOR model

    labels=tf.ones((batch_size,1))
    # generator wants discriminator to think fake images as real
    with tf.GradientTape() as tape:
      # generate fake images from generator
      fake_images=self.generator(random_noise,training=True)
      # classify images as real or fake
      pred_fake=self.discriminator(fake_images,training=True)
      # compute loss
      g_loss=self.loss_fn(labels,pred_fake)

    # compute gradients
    gradients=tape.gradient(g_loss,self.generator.trainable_variables)
    # update the gradients
    self.g_optimizer.apply_gradients(zip(gradients,self.generator.trainable_variables))

    # update states for both models
    self.g_loss_metric.update_state(g_loss)
    self.d_loss_metric.update_state(d_loss)

    return {"g_loss":self.g_loss_metric.result(),"d_loss":self.d_loss_metric.result()}

class DCGANMonitor(keras.callbacks.Callback):
  def __init__(self,num_img=20,latent_dim=100):
    self.num_img=num_img
    self.latent_dim=latent_dim

    # create random noise for generating images
    self.noise=tf.random.normal(shape=(self.num_img,self.latent_dim))

  def on_epoch_end(self,epoch,logs=None):
    # generating image from noise
    generated_images=self.model.generator(self.noise)
    # denoising the image
    generated_images= (generated_images*127.5)+127.5
    # generated_images=array_to_img(generated_images)

    # plot the generated images
    fig=plt.figure(figsize=(8,8))
    for i in range(self.num_img):
      plt.subplot(5,5,i+1)
      img= array_to_img(generated_images[i])
      plt.imshow(img)
      plt.axis("off")

    # plt.savefig('epoch_{:03d}.png'.format(epoch))
    plt.show()

    def on_train_end(self,logs=None):
      self.model.generator.save("generator.h5")

dcgan=DCGAN(discriminator=discriminator,generator=generator,latent_dim=latent_dim)

D_LR=0.001
G_LR=0.003
dcgan.compile(g_optimizer=Adam(learning_rate=G_LR,beta_1=0.5),d_optimizer=Adam(learning_rate=D_LR,beta_1=0.5),loss_fn=BinaryCrossentropy())

# Define the checkpoint filepath
checkpoint_filepath = "model_checkpoint.weights.h5"

# Create a ModelCheckpoint callback
checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_filepath,  # Path to save the checkpoint
    save_weights_only=True,  # Save only the weights, not the entire model
    save_best_only=False,  # Save weights every epoch, not just the best ones
    save_freq='epoch'  # Frequency of saving: at the end of every epoch
)

# Load weights if checkpoint exists
if os.path.exists(checkpoint_filepath):  # Check if the checkpoint file exists
    dcgan.load_weights(checkpoint_filepath)  # Load weights into the model
    print("Resuming training from checkpoint...")
else:
    print("Starting training from scratch...")

# Before training (to resume):
if os.path.exists('train_history.json'):
    with open('train_history.json', 'r') as f:
        history = json.load(f)

    # Plotting the loss curves
    plt.figure(figsize=(10, 5))
    plt.plot(history['g_loss'], label='Generator Loss')
    plt.plot(history['d_loss'], label='Discriminator Loss')
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

# Build the model explicitly before training or loading weights
dcgan.build(input_shape=(None, 64, 64, 3))

N_EPOCHS=50
history = dcgan.fit(train_images,batch_size=32,epochs=N_EPOCHS,callbacks=[DCGANMonitor(latent_dim=latent_dim),checkpoint_callback]).history
