# example of a dcgan on cifar10
import numpy as np
from tensorflow.keras.datasets.cifar10 import load_data
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Dropout
from matplotlib import pyplot as plt
 
# define the standalone discriminator model
def define_discriminator(in_shape=(32,32,3)):
	model = Sequential()
	# normal
	model.add(Conv2D(64, (3,3), padding='same', input_shape=in_shape))
	model.add(LeakyReLU(alpha=0.2))
	# downsample
	model.add(Conv2D(128, (3,3), strides=(2,2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	# downsample
	model.add(Conv2D(128, (3,3), strides=(2,2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	# downsample
	model.add(Conv2D(256, (3,3), strides=(2,2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	# classifier
	model.add(Flatten())
	model.add(Dropout(0.4))
	model.add(Dense(1, activation='sigmoid'))
	# compile model
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
	return model
 
# define the standalone generator model
def define_generator(latent_dim):
	model = Sequential()
	# foundation for 4x4 image
	n_nodes = 256 * 4 * 4
	model.add(Dense(n_nodes, input_dim=latent_dim))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Reshape((4, 4, 256)))
	# upsample to 8x8
	model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	# upsample to 16x16
	model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	# upsample to 32x32
	model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	# output layer
	model.add(Conv2D(3, (3,3), activation='tanh', padding='same'))
	return model
 
# define the combined generator and discriminator model, for updating the generator
def define_gan(g_model, d_model):
	# make weights in the discriminator not trainable
	d_model.trainable = False
	# connect them
	model = Sequential()
	# add generator
	model.add(g_model)
	# add the discriminator
	model.add(d_model)
	# compile model
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt)
	return model
 
# load and prepare cifar10 training images
def load_real_samples():
	# load cifar10 dataset
	(trainX, _), (_, _) = load_data()
	# convert from unsigned ints to floats
	X = trainX.astype('float32')
	# scale from [0,255] to [-1,1]
	X = (X - 127.5) / 127.5
	return X
 
# select real samples
def generate_real_samples(dataset, n_samples):
	# choose random instances
	ix = np.random.randint(0, dataset.shape[0], n_samples)
	# retrieve selected images
	X = dataset[ix]
	# generate 'real' class labels (1)
	y = np.ones((n_samples, 1))
	return X, y
 
# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples):
	# generate points in the latent space
	x_input = np.random.randn(latent_dim * n_samples)
	# reshape into a batch of inputs for the network
	x_input = x_input.reshape(n_samples, latent_dim)
	return x_input
 
# use the generator to generate n fake examples, with class labels
def generate_fake_samples(g_model, latent_dim, n_samples):
	# generate points in latent space
	x_input = generate_latent_points(latent_dim, n_samples)
	# predict outputs
	X = g_model.predict(x_input)
	# create 'fake' class labels (0)
	y = np.zeros((n_samples, 1))
	return X, y
 
# create and save a plot of generated images
def save_plot(examples, epoch, n=7):
	# scale from [-1,1] to [0,1]
	examples = (examples + 1) / 2.0
	# plot images
	for i in range(n * n):
		# define subplot
		plt.subplot(n, n, 1 + i)
		# turn off axis
		plt.axis('off')
		# plot raw pixel data
		plt.imshow(examples[i])
	# save plot to file
	filename = 'generated_plot_e%03d.png' % (epoch+1)
	plt.savefig(filename)
	plt.close()
 
# evaluate the discriminator, plot generated images, save generator model
def summarize_performance(epoch, g_model, d_model, dataset, latent_dim, n_samples=150):
	# prepare real samples
	X_real, y_real = generate_real_samples(dataset, n_samples)
	# evaluate discriminator on real examples
	_, acc_real = d_model.evaluate(X_real, y_real, verbose=0)
	# prepare fake examples
	x_fake, y_fake = generate_fake_samples(g_model, latent_dim, n_samples)
	# evaluate discriminator on fake examples
	_, acc_fake = d_model.evaluate(x_fake, y_fake, verbose=0)
	# summarize discriminator performance
	print('>Accuracy real: %.0f%%, fake: %.0f%%' % (acc_real*100, acc_fake*100))
	# save plot
	save_plot(x_fake, epoch)
	# save the generator model tile file
	filename = 'generator_model_%03d.h5' % (epoch+1)
	g_model.save(filename)
 
# train the generator and discriminator
def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=200, n_batch=128):
	bat_per_epo = int(dataset.shape[0] / n_batch)
	half_batch = int(n_batch / 2)
	# manually enumerate epochs
	for i in range(n_epochs):
		# enumerate batches over the training set
		for j in range(bat_per_epo):
			# get randomly selected 'real' samples
			X_real, y_real = generate_real_samples(dataset, half_batch)
			# update discriminator model weights
			d_loss1, _ = d_model.train_on_batch(X_real, y_real)
			# generate 'fake' examples
			X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
			# update discriminator model weights
			d_loss2, _ = d_model.train_on_batch(X_fake, y_fake)
			# prepare points in latent space as input for the generator
			X_gan = generate_latent_points(latent_dim, n_batch)
			# create inverted labels for the fake samples
			y_gan = np.ones((n_batch, 1))
			# update the generator via the discriminator's error
			g_loss = gan_model.train_on_batch(X_gan, y_gan)
			# summarize loss on this batch
			print('>epoch %d, example %d/%d, d1=%.3f, d2=%.3f g=%.3f' %
				(i+1, j+1, bat_per_epo, d_loss1, d_loss2, g_loss))
		# evaluate the model performance, sometimes
		if (i+1) % 10 == 0:
			summarize_performance(i, g_model, d_model, dataset, latent_dim)
 
# size of the latent space
latent_dim = 100
# create the discriminator
d_model = define_discriminator()
# create the generator
g_model = define_generator(latent_dim)
# create the gan
gan_model = define_gan(g_model, d_model)
# load image data
dataset = load_real_samples()
# train model
train(g_model, d_model, gan_model, dataset, latent_dim,epochs=5)




"""
import numpy as np 
import matplotlib.pyplot as plt 
import tensorflow.keras as keras
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout 
from tensorflow.keras.layers import BatchNormalization, Activation, ZeroPadding2D, LeakyReLU
from tensorflow.keras.layers import UpSampling2D, Conv2D 
from tensorflow.keras.models import Sequential, Model 
from tensorflow.keras.optimizers import Adam,SGD 

#Loading the CIFAR10 data 
(X, y), (_, _) = keras.datasets.cifar10.load_data() 

#Selecting a single class images 
#The number was randomly chosen and any number 
#between 1 to 10 can be chosen 
X = X[y.flatten() == 8] 

#Defining the Input shape 
image_shape = (32, 32, 3) 
		
latent_dimensions = 100

def build_generator(): 

		model = Sequential() 

		#Building the input layer 
		model.add(Dense(128 * 8 * 8, activation="relu", 
						input_dim=latent_dimensions)) 
		model.add(Reshape((8, 8, 128))) 
		
		model.add(UpSampling2D()) 
		
		model.add(Conv2D(128, kernel_size=3, padding="same")) 
		model.add(BatchNormalization(momentum=0.78)) 
		model.add(Activation("relu")) 
		
		model.add(UpSampling2D()) 
		
		model.add(Conv2D(64, kernel_size=3, padding="same")) 
		model.add(BatchNormalization(momentum=0.78)) 
		model.add(Activation("relu")) 
		
		model.add(Conv2D(3, kernel_size=3, padding="same")) 
		model.add(Activation("tanh")) 


		#Generating the output image 
		noise = Input(shape=(latent_dimensions,)) 
		image = model(noise) 

		return Model(noise, image) 

def build_discriminator(): 

		#Building the convolutional layers 
		#to classify whether an image is real or fake 
		model = Sequential() 

		model.add(Conv2D(32, kernel_size=3, strides=2, 
						input_shape=image_shape, padding="same")) 
		model.add(LeakyReLU(alpha=0.2)) 
		model.add(Dropout(0.25)) 
		
		model.add(Conv2D(64, kernel_size=3, strides=2, padding="same")) 
		model.add(ZeroPadding2D(padding=((0,1),(0,1)))) 
		model.add(BatchNormalization(momentum=0.82)) 
		model.add(LeakyReLU(alpha=0.25)) 
		model.add(Dropout(0.25)) 
		
		model.add(Conv2D(128, kernel_size=3, strides=2, padding="same")) 
		model.add(BatchNormalization(momentum=0.82)) 
		model.add(LeakyReLU(alpha=0.2)) 
		model.add(Dropout(0.25)) 
		
		model.add(Conv2D(256, kernel_size=3, strides=1, padding="same")) 
		model.add(BatchNormalization(momentum=0.8)) 
		model.add(LeakyReLU(alpha=0.25)) 
		model.add(Dropout(0.25)) 
		
		#Building the output layer 
		model.add(Flatten()) 
		model.add(Dense(1, activation='sigmoid')) 

		image = Input(shape=image_shape) 
		validity = model(image) 

		return Model(image, validity) 

def display_images(): 
		r, c = 4,4
		noise = np.random.normal(0, 1, (r * c,latent_dimensions)) 
		generated_images = generator.predict(noise) 

		#Scaling the generated images 
		generated_images = 0.5 * generated_images + 0.5

		fig, axs = plt.subplots(r, c) 
		count = 0
		for i in range(r): 
			for j in range(c): 
				axs[i,j].imshow(generated_images[count, :,:,]) 
				axs[i,j].axis('off') 
				count += 1
		plt.show() 
		plt.close() 


# Building and compiling the discriminator 
discriminator = build_discriminator() 
discriminator.compile(loss='binary_crossentropy', 
					optimizer=Adam(0.0002,0.5), 
					metrics=['accuracy']) 

#Making the Discriminator untrainable 
#so that the generator can learn from fixed gradient 
discriminator.trainable = False

# Building the generator 
generator = build_generator() 

#Defining the input for the generator 
#and generating the images 
z = Input(shape=(latent_dimensions,)) 
image = generator(z) 


#Checking the validity of the generated image 
valid = discriminator(image) 

#Defining the combined model of the Generator and the Discriminator 
combined_network = Model(z, valid) 
combined_network.compile(loss='binary_crossentropy', 
						optimizer=Adam(0.0002,0.5)) 

num_epochs=15000
batch_size=32
display_interval=2500
losses=[] 

#Normalizing the input 
X = (X / 127.5) - 1.
		

#Defining the Adversarial ground truths 
valid = np.ones((batch_size, 1)) 

#Adding some noise 
valid += 0.05 * np.random.random(valid.shape) 
fake = np.zeros((batch_size, 1)) 
fake += 0.05 * np.random.random(fake.shape) 

for epoch in range(num_epochs): 
			
			#Training the Discriminator 
			
			#Sampling a random half of images 
			index = np.random.randint(0, X.shape[0], batch_size) 
			images = X[index] 

			#Sampling noise and generating a batch of new images 
			noise = np.random.normal(0, 1, (batch_size, latent_dimensions)) 
			generated_images = generator.predict(noise) 
			

			#Training the discriminator to detect more accurately 
			#whether a generated image is real or fake 
			discm_loss_real = discriminator.train_on_batch(images, valid) 
			discm_loss_fake = discriminator.train_on_batch(generated_images, fake) 
			discm_loss = 0.5 * np.add(discm_loss_real, discm_loss_fake) 
			
			#Training the Generator 

			#Training the generator to generate images 
			#which pass the authenticity test 
			genr_loss = combined_network.train_on_batch(noise, valid) 
			
			#Tracking the progress				 
			if epoch % display_interval == 0: 
				display_images() 


#Plotting some of the original images 
s=X[:40] 
s = 0.5 * s + 0.5
f, ax = plt.subplots(5,8, figsize=(16,10)) 
for i, image in enumerate(s): 
	ax[i//8, i%8].imshow(image) 
	ax[i//8, i%8].axis('off') 
		
plt.show() 

#Plotting some of the last batch of generated images 
noise = np.random.normal(size=(40, latent_dimensions)) 
generated_images = generator.predict(noise) 
generated_images = 0.5 * generated_images + 0.5
f, ax = plt.subplots(5,8, figsize=(16,10)) 
for i, image in enumerate(generated_images): 
	ax[i//8, i%8].imshow(image) 
	ax[i//8, i%8].axis('off') 
		
plt.show() 
"""
