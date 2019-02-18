import tensorflow as tf
import numpy as np
import general_constants as gc
import general_utils as gu
import os
import matplotlib.pyplot as plt
from functools import reduce
import scipy as scp
import shutil
import params as pm


#create the encoder:
def encoder(pred, activation=tf.nn.relu, latent_size=32):
	with tf.variable_scope("encoder"):
		#we will run this architecture later on, mnist is too small to run these
		#pred = tf.layers.conv2d(pred, 32, 4, 2, "same", activation=activation)
		#pred = tf.layers.conv2d(pred, 32, 4, 2, "same", activation=activation)
		pred = tf.layers.conv2d(pred, 64, 4, 2, "same", activation=activation)
		pred = tf.layers.conv2d(pred, 64, 4, 2, "same", activation=activation)
		shape_before_flatten = tf.shape(pred)

		pred = tf.contrib.layers.flatten(pred)
		pred = tf.layers.dense(pred,256)
		
		latent_mean = tf.layers.dense(pred, latent_size)

		#if we use the log, we can be negative
		latent_log_std = tf.layers.dense(pred, latent_size)
		noise = tf.random_normal(tf.shape(latent_log_std))
		latent = tf.exp(latent_log_std)*noise+latent_mean
	return latent, shape_before_flatten, [latent_mean, latent_log_std]


#create the decoder:
def decoder(latent_rep, shape_before_flatten, activation=tf.nn.relu):
	with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):
		pred = tf.layers.dense(latent_rep, 256)
		pred = tf.layers.dense(pred, 7*7*64)#this is hard coded right now
		pred = tf.reshape(pred, shape_before_flatten)

		pred = tf.layers.conv2d_transpose(pred, 64, 4, 2, "same", activation=activation)
		
		#we will run this architecture later on, mnist is too small to run these
		#pred = tf.layers.conv2d(pred, 64, 4, 2, "same", activation=activation)
		#pred = tf.layers.conv2d(pred, 32, 4, 2, "same", activation=activation)
		pred = tf.layers.conv2d_transpose(pred, 1, 4, 2, "same", activation=activation)

		#compress values to be between 0 and 1
		#pred = tf.sigmoid(pred)
		
	return pred

def kl_isonormal_loss(mu, sig):
	#finds the KL divergence between an entered function and an isotropic normal
	loss = tf.reduce_sum(tf.exp(sig),axis=1)
	loss = loss+tf.reduce_sum(tf.square(mu))
	loss = loss-tf.cast(tf.shape(mu)[-1], tf.float32)
	loss = loss-tf.reduce_sum(sig,axis=1)
	loss = 0.5*(loss/tf.cast(tf.shape(mu)[-1], tf.float32))

	return loss
def reconstruction_loss(inputs, pred, loss_type=gu.cross_entropy):
	#return tf.losses.mean_squared_error(inputs, pred)
	#loss_type is a function of the loss to apply. sould only take inputs and pred
	return loss_type(inputs,pred)

def main():
	params = pm.model_params
	batch_size = params["batch_size"]
	log_step = params["log_step"]
	num_steps = params["num_steps"]
	learning_rate = params["learning_rate"]
	plot_step = params["plot_step"]
	loss_type = params["loss_type"]
	latent_size = params["latent_size"]
	#get data
	images, labels = gu.get_mnist_data(gc.datapath)
	datashape = list(images.shape)
	datashape[0] = batch_size

	#create placeholders
	inputs_ph = tf.placeholder(tf.float32, shape=(None, *datashape[1:]))
	outputs_ph = tf.placeholder(tf.float32)
	inputs_set_ph = tf.placeholder(tf.float32)
	outputs_set_ph = tf.placeholder(tf.float32)	
	iterator, next_element = gu.get_iterator(batch_size, inputs=inputs_set_ph, labels=outputs_set_ph)

	#make model
	inputs = inputs_ph
	pred, shape_before_flatten, dist_params = encoder(inputs, latent_size=latent_size)
	pred = decoder(pred, shape_before_flatten)

	#get loss
	Recon_Loss = tf.reduce_mean(reconstruction_loss(inputs, pred, loss_type))
	Regul_Loss = tf.reduce_mean(kl_isonormal_loss(*dist_params))
	kl_multiplier = tf.placeholder(tf.float32)
	loss = Recon_Loss+kl_multiplier*Regul_Loss

	#training:
	opt = tf.train.AdamOptimizer(learning_rate)
	minim = opt.minimize(loss)
	with tf.control_dependencies([minim]):
		train_op = tf.no_op()

	#run model
	with tf.Session() as sess:
		#print(training_data["data"].shape)
		sess.run(tf.global_variables_initializer())
		sess.run(iterator.initializer, feed_dict={
				inputs_set_ph:images,
				outputs_set_ph:labels,
			})
		for step in range(num_steps):
			data = sess.run(next_element)
			feed_dict = {
				inputs_ph:data["inputs"], 
				outputs_ph:data["labels"],
				kl_multiplier:1
				}


			loss_val, Regul_Loss_val, Recon_Loss_val = sess.run([loss, Regul_Loss, Recon_Loss], feed_dict=feed_dict)
			
			sess.run([train_op], feed_dict=feed_dict)

			print("step: %d, \ttotal loss: %.3f, \tRegularization loss: %.3f, \tReconstruction loss: %.3f"%(step, loss_val, Regul_Loss_val, Recon_Loss_val))
			if np.isnan(loss_val):
				break
			if not step%plot_step or step in log_step:
				#save image of data:
				data_val = sess.run(pred, feed_dict=feed_dict)
				subplot = [2,3]
				f, axarr = plt.subplots(*subplot)
				for i in range(np.prod(subplot).item()):
					axarr[i%subplot[0],i//subplot[0]%subplot[1]].imshow(data_val[i,:,:,0])
				plt.savefig(os.path.join(pm.logdir, "image_%s.jpg"%step))
				plt.close()


if __name__ == "__main__":
	main()