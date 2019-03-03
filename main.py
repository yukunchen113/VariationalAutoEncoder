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
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
"""
#create the encoder:
def encoder(pred, activation=tf.nn.relu, latent_size=32):
	with tf.variable_scope("encoder"):
		#we will run this architecture later on, mnist is too small to run these
		
		pred = tf.layers.conv2d(pred, 64, 4, 2, "same", activation=activation)
		pred = tf.layers.conv2d(pred, 64, 4, 2, "same", activation=activation)
		shape_before_flatten = tf.shape(pred)

		pred = tf.contrib.layers.flatten(pred)
		pred = tf.layers.dense(pred,256)
		pred = activation(pred)

		latent_mean = tf.layers.dense(pred, latent_size)
		#if we use the log, we can be negative
		latent_log_var = tf.layers.dense(pred, latent_size)
		noise = tf.random_normal(tf.shape(latent_log_var))
		latent = tf.exp(0.5*latent_log_var)*noise+latent_mean
	return latent, shape_before_flatten, [latent_mean, latent_log_var]


#create the decoder:
def decoder(latent_rep, shape_before_flatten, activation=tf.nn.relu):
	with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):
		pred = tf.layers.dense(latent_rep, 256)
		pred = activation(pred)
		pred = tf.layers.dense(pred, 7*7*64)#this is hard coded right now
		pred = tf.reshape(pred, shape_before_flatten)

		pred = tf.layers.conv2d_transpose(pred, 64, 4, 2, "same", activation=activation)
		
		#we will run this architecture later on, mnist is too small to run these
		pred = tf.layers.conv2d_transpose(pred, 1, 4, 2, "same", activation=activation)

		#compress values to be between 0 and 1
		pred = tf.sigmoid(pred)
		
	return pred
"""
#create the encoder:
def encoder(pred, activation=tf.nn.relu, latent_size=32):
	with tf.variable_scope("encoder"):
		#we will run this architecture later on, mnist is too small to run these
		shape_before_flatten = tf.shape(pred)
		pred = tf.contrib.layers.flatten(pred)

		# 1st hidden layer
		pred = tf.contrib.layers.fully_connected(
			pred,
			500
			)

		# 2nd hidden layer
		pred = tf.contrib.layers.fully_connected(
			pred,
			500
			)


		latent_mean = tf.layers.dense(pred, latent_size)
		#if we use the log, we can be negative
		latent_log_var = tf.layers.dense(pred, latent_size)
		noise = tf.random_normal(tf.shape(latent_log_var))
		latent = tf.exp(0.5*latent_log_var)*noise+latent_mean
	return latent, shape_before_flatten, [latent_mean, latent_log_var]


#create the decoder:
def decoder(latent_rep, shape_before_flatten, activation=tf.nn.relu):
	with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):

		# 1st hidden layer
		pred = tf.contrib.layers.fully_connected(
			latent_rep,
			500
			)

		# output layer-mean
		pred = tf.contrib.layers.fully_connected(
			pred,
			784,
			activation_fn=tf.nn.sigmoid
			)


		pred = tf.reshape(pred, shape_before_flatten)
		
	return pred

def kl_isonormal_loss(mu, logvar):
	#finds the KL divergence between an entered function and an isotropic normal
	loss = gu.kl_divergence(mu, logvar)
	return loss
def reconstruction_loss(inputs, pred, loss_type=gu.cross_entropy):
	#return tf.losses.mean_squared_error(inputs, pred)
	#loss_type is a function of the loss to apply. sould only take inputs and pred
	return loss_type(tf.contrib.layers.flatten(inputs),tf.contrib.layers.flatten(pred))

def main():
	params = pm.model_params
	batch_size = params["batch_size"]
	log_step = params["log_step"]
	num_steps = params["num_steps"]
	learning_rate = params["learning_rate"]
	plot_step = params["plot_step"]
	loss_type = params["loss_type"]
	latent_size = params["latent_size"]
	log_file = "log.txt" 
	if os.path.exists(log_file):
		os.remove(log_file)
	pm.create_new_path(pm.logdir)
	#get data
	images, labels = gu.get_mnist_data(gc.datapath)
	datashape = list(images.shape)
	datashape[0] = batch_size

	#create placeholders
	inputs_ph = tf.placeholder(tf.float32, shape=(None, *datashape[1:]), name="inputs_ph")
	outputs_ph = tf.placeholder(tf.float32, name="outputs_ph")
	inputs_set_ph = tf.placeholder(tf.float32, name="inputs_set_ph")
	outputs_set_ph = tf.placeholder(tf.float32, name="outputs_set_ph")
	latents_ph = tf.placeholder(tf.float32, shape=(None, latent_size), name="latents_ph")
	iterator, next_element = gu.get_iterator(batch_size, inputs=inputs_set_ph, labels=outputs_set_ph)

	#make model
	inputs = inputs_ph
	latents_gen = latents_ph
	latents_rec, shape_before_flatten, dist_params = encoder(inputs, latent_size=latent_size)
	pred_rec = decoder(latents_rec, shape_before_flatten)#decoder for reconstruction
	pred_gen = decoder(latents_gen, shape_before_flatten)#decoder for generation of new samples 

	#get loss
	Recon_Loss = tf.reduce_mean(reconstruction_loss(inputs, pred_rec, loss_type))
	Regul_Loss = tf.reduce_mean(kl_isonormal_loss(*dist_params))
	kl_multiplier = tf.placeholder(tf.float32, name="kl_multiplier")
	loss = Recon_Loss+kl_multiplier*Regul_Loss

	#training:
	train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)


	#latent space analysis:
	std_analysis = [tf.reduce_mean(tf.exp(0.5*dist_params[1])), tf.reduce_min(tf.exp(0.5*dist_params[1])), tf.reduce_max(tf.exp(0.5*dist_params[1]))]
	mean_analysis = [tf.reduce_mean(dist_params[0]), tf.reduce_min(dist_params[0]), tf.reduce_max(dist_params[0])]


	#run model
	with tf.Session() as sess:
		#print(training_data["data"].shape)
		sess.run(tf.global_variables_initializer())
		test_images = images[:batch_size]
		test_feed_dict = {#keep all these images to test constant.
			inputs_ph:test_images, 
			outputs_ph:labels[:batch_size],
			latents_ph:np.random.normal(size=(batch_size, latent_size))
		}
		sess.run(iterator.initializer, feed_dict={
				inputs_set_ph:images,
				outputs_set_ph:labels,
			})
		kl_mul_val_step = 0#initial multiplier value
		#recon_bound = 0.022
		#recon_static= 0#number of steps where the KLD weight is not increasing
		activate = False#this will start the KLD weight, only when the reconstruction has first reached oast the bound.
		with open(log_file, "a") as log_file:
			for step in range(num_steps):
				data = sess.run(next_element)
				feed_dict = {
					inputs_ph:data["inputs"], 
					outputs_ph:data["labels"],
					}


				Regul_Loss_val, Recon_Loss_val = sess.run([Regul_Loss, Recon_Loss], feed_dict=feed_dict)
				
				train_feed = feed_dict.copy()
				
				if step>3000:
					activate = True

				if activate:
					kl_mul_val_step += 1/50000
					kl_mul_val_step = min(1, max(0, kl_mul_val_step))
				
				
				train_feed[kl_multiplier] = 1#kl_mul_val_step#min((step - kl_mul_val_step)/30000, 1)
					

				loss_val,_ = sess.run([loss, train_op], feed_dict=train_feed)

				print_out = "step: %d, \ttotal loss: %.3f, \tRegularization loss: %.3f, \tReconstruction loss: %.3f, kl weight: %f,\
				\n Latent Space Analysis: \naverage stddev %s,\t stddev range [%s,	 \t%s], \
				\naverage mean %s,   \tmean range [%s,	\t%s]"%(step, loss_val, Regul_Loss_val, Recon_Loss_val, train_feed[kl_multiplier], *sess.run([*std_analysis, *mean_analysis], feed_dict=feed_dict))
				print(print_out)
				print(""%sess.run([], feed_dict=feed_dict))
				log_file.write("%s\n"%print_out)
				if np.isnan(loss_val):
					break
				
				if not step%plot_step or step in log_step:
					#save image of data:
					#create reconstruction
					recon_val, gener_val = sess.run([pred_rec, pred_gen], feed_dict=test_feed_dict)

					original_images = gu.create_image_grid(test_images[:48], [1,9])
					reconstruction = gu.create_image_grid(recon_val[:48],[1,9])
					generation = gu.create_image_grid(gener_val[:48],[1,9])
					

					#create images
					captions = ["reconstruction original images", "image reconstruction", "image generation"]
					images_type = [original_images, reconstruction, generation]
					im = []
					header_size = 30#amount of space for caption
					for i in range(len(images_type)):
						image_type = np.log(5*(images_type[i]+1))
						image_type = (image_type-np.amin(image_type))
						image_type = image_type/np.amax(image_type)
						caption = captions[i]
						container = np.ones((image_type.shape[0]+header_size, image_type.shape[1]))
						container[:-header_size] = image_type
						im.append(Image.fromarray(np.uint8(container*255)))
						ImageDraw.Draw(im[i]).text((5,image_type.shape[0]+2), caption)
					
					width = max([i.size[0] for i in im])
					height = sum([i.size[1] for i in im])
					header = 40
					margin = 20
					total_image = Image.fromarray(np.ones((height+header, width+margin))*255)
					for i in range(len(im)):
						image = im[i]
						total_image.paste(image, (margin//2,header+i*image.size[1]))
					ImageDraw.Draw(total_image).text((margin//2+10,5), "Step: %d"%step)
					total_image.convert('RGB').save(os.path.join(pm.logdir, "image_%s.jpg"%step))
					"""
					plt.clf()
					subplot = [3,1]
					f, axarr = plt.subplots(*subplot, constrained_layout=True)
					axarr[0].imshow(original_images)
					axarr[0].set_title("reconstruction original images")
					axarr[1].imshow(reconstruction)
					axarr[1].set_title("image reconstruction, Step: %d"%step)
					axarr[2].imshow(generation)
					axarr[2].set_title("image generation, Step: %d"%step)
					plt.savefig(os.path.join(pm.logdir, "image_%s.jpg"%step))
					plt.close()
					np.savez(os.path.join(pm.logdir, "image_%s.npz"%step),
					 **{"original_images":original_images, 
					 "reconstruction":reconstruction, 
					 "generation":generation})
					"""
				

if __name__ == "__main__":
	main()