import tensorflow as tf
import numpy as np
import datetime

###############################################
###############################################

data_dir = '/lustre05/vol0/kemuwz/ml/fpc/extraction/data/lower3/' 
model_number = 10000   #which saved model to load if restarting training 


try:
  xt = np.load(data_dir+'xt.npy')  # load training data 
  xv = np.load(data_dir+'xv.npy')  # load val data 
except exception as e:
  print(e)
  print('problem loading data')

# train/val data is numpy array of shape (nsamples, 256, 256, 3)
# consists of normalized u,v,p values on a regular structured 256x256 grid, from snapshots of cfd simualtion at fixed intervals

ntrain = xt.shape[0]
nval = xv.shape[0]
w = xt.shape[1]
h = xt.shape[2]
ndims = xt.shape[3]

batch_size = 16

batches_train = int((ntrain)/batch_size)
batches_validate = int((nval)/batch_size)

xv2 = xv.reshape(batches_validate, batch_size, w, h, ndims)


###############################################
###############################################

#make encoder network
def make_encoder():     
  encoder = tf.keras.Sequential()

  # convolutional layers
  encoder.add(tf.keras.layers.Conv2D(6, kernel_size=3, input_shape=(w, h, ndims)))
  encoder.add(tf.keras.layers.LeakyReLU(alpha=0.3))
  encoder.add(tf.keras.layers.Conv2D(12, kernel_size=3, strides=(2, 2)))
  encoder.add(tf.keras.layers.LeakyReLU(alpha=0.3))
  encoder.add(tf.keras.layers.Conv2D(24, kernel_size=3, strides=(2, 2)))
  encoder.add(tf.keras.layers.LeakyReLU(alpha=0.3))
  encoder.add(tf.keras.layers.Conv2D(48, kernel_size=3, strides=(2, 2)))
  encoder.add(tf.keras.layers.LeakyReLU(alpha=0.3))
  encoder.add(tf.keras.layers.Conv2D(96, kernel_size=3, strides=(2, 2)))
  encoder.add(tf.keras.layers.LeakyReLU(alpha=0.3))
  encoder.add(tf.keras.layers.Conv2D(192, kernel_size=3, strides=(2, 2)))
  encoder.add(tf.keras.layers.LeakyReLU(alpha=0.3))
  encoder.add(tf.keras.layers.Flatten())

  # fully connected layers
  encoder.add(tf.keras.layers.Dense(512))
  encoder.add(tf.keras.layers.LeakyReLU(alpha=0.3))
  encoder.add(tf.keras.layers.Dense(128, activation='tanh'))

  return encoder

#make decoder network
def make_decoder():
  decoder = tf.keras.Sequential()

  #fully connected layers
  decoder.add(tf.keras.layers.Dense(512, input_shape=(128,)))
  decoder.add(tf.keras.layers.LeakyReLU(alpha=0.3))
  decoder.add(tf.keras.layers.Dense(6912))
  decoder.add(tf.keras.layers.LeakyReLU(alpha=0.3))
  decoder.add(tf.keras.layers.Reshape((6, 6, 192)))

  # convolutional layers
  decoder.add(tf.keras.layers.Conv2DTranspose(96, kernel_size=4, strides=(2, 2)))
  decoder.add(tf.keras.layers.LeakyReLU(alpha=0.3))
  decoder.add(tf.keras.layers.Conv2DTranspose(48, kernel_size=4, strides=(2, 2)))
  decoder.add(tf.keras.layers.LeakyReLU(alpha=0.3))
  decoder.add(tf.keras.layers.Conv2DTranspose(24, kernel_size=4, strides=(2, 2)))
  decoder.add(tf.keras.layers.LeakyReLU(alpha=0.3))
  decoder.add(tf.keras.layers.Conv2DTranspose(12, kernel_size=4, strides=(2, 2)))
  decoder.add(tf.keras.layers.LeakyReLU(alpha=0.3))
  decoder.add(tf.keras.layers.Conv2DTranspose(6, kernel_size=4, strides=(2, 2)))
  decoder.add(tf.keras.layers.LeakyReLU(alpha=0.3))
  decoder.add(tf.keras.layers.Conv2DTranspose(3, kernel_size=3, activation='linear'))

  return decoder


try: 
  #if continuing training from checkpoint
  saved_enc_dir = './saved_encoder_' + str(model_number)
  saved_dec_dir = './saved_decoder_' + str(model_number)
  encoder = tf.keras.models.load_model(saved_enc_dir)
  decoder = tf.keras.models.load_model(saved_dec_dir)
  print('loaded model')
except:  
  #if starting training from scratch
  encoder = make_encoder()
  decoder = make_decoder()

mse = tf.keras.losses.MeanSquaredError()

def ae_loss(inp, outp):
    return mse(inp, outp)

# network optimizer
encoder_optimizer = tf.keras.optimizers.Adamax(1e-4)
decoder_optimizer = tf.keras.optimizers.Adamax(1e-4)

#losses for tracking during training
train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
val_loss = tf.keras.metrics.Mean('val_loss', dtype=tf.float32)


###############################################
###############################################

#training step
#use @tf.function to compile as graph for faster training

@tf.function
def train_autoencoder_step(X_train):
    images_in = X_train # training data

    with tf.GradientTape() as enc_tape, tf.GradientTape() as dec_tape:
      compressed = encoder(images_in, training=True) # compress data
      uncompressed = decoder(compressed, training=True) # reconstruct data
      autoencoder_loss = ae_loss(images_in, uncompressed) # find mse loss between reconstructed and real data

    gradients_of_encoder = enc_tape.gradient(autoencoder_loss, encoder.trainable_variables) # find gradients wrt to loss for encoder
    gradients_of_decoder = dec_tape.gradient(autoencoder_loss, decoder.trainable_variables) # find gradients wrt to loss for decoder

    encoder_optimizer.apply_gradients(zip(gradients_of_encoder, encoder.trainable_variables)) # optimize encoder weights
    decoder_optimizer.apply_gradients(zip(gradients_of_decoder, decoder.trainable_variables)) # optimize decoder weights

    train_loss(autoencoder_loss) # record loss for tracking

#validation step
@tf.function
def val_step(X_val):
    images_val = X_val

    compressed_val = encoder(images_val, training=False)
    uncompressed_val = decoder(compressed_val, training=False) 

    validation_loss = ae_loss(images_val, uncompressed_val)

    val_loss(validation_loss)


# create tensorboard tracking 
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = '../logs/gradient_tape/' + current_time + '/train'
val_log_dir = '../logs/gradient_tape/' + current_time + '/val'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
val_summary_writer = tf.summary.create_file_writer(val_log_dir)


def train(X_train, X_val, epochs):
  Xt = X_train
  np.random.shuffle(Xt)
  X_t = Xt.reshape(batches_train, batch_size, w, h, 3)

  for epoch in range(epochs):
    X_v = X_val
    
    if (epoch + 1) % 10 == 0:
        Xt = X_train
        np.random.shuffle(Xt)
        X_t = Xt.reshape(batches_train, batch_size, w, h, 3)

    for i in range(len(X_t)):
      train_autoencoder_step(X_t[i])
    
    for i in range(len(X_v)):
      val_step(X_v[i])

    with train_summary_writer.as_default():
      tf.summary.scalar('loss', train_loss.result(), step=epoch)

    with val_summary_writer.as_default():
      tf.summary.scalar('loss', val_loss.result(), step=epoch)
      

    if (epoch + 1) % 100 == 0:
      saved_enc_dir = './saved_encoder_' + str(epoch + 1)
      saved_dec_dir = './saved_decoder_' + str(epoch + 1)

      tf.keras.models.save_model(encoder, saved_enc_dir)
      tf.keras.models.save_model(decoder, saved_dec_dir)

    train_loss.reset_states()
    val_loss.reset_states()
      

EPOCHS = 10000

train(xt, xv2, EPOCHS) 
