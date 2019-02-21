import os
import general_utils as gu
import tensorflow as tf
import shutil
#set paths
logdir = "predictions"
tblogdir = "tensorboard_logdir"
def create_new_path(path):
	if not os.path.exists(path):
		os.mkdir(path)
	else:
		shutil.rmtree(path)
		os.mkdir(path)

model_params = {
	#model params
	"batch_size":64,#the size for batch training
	"latent_size":32,

	#for plotting/logging
	"plot_step":1000,#will save the plot every this many steps.
	"log_step":[0,1,3,7,10,16,30,100,300,500,700],#additional plot saves on these steps
	
	#for training
	"learning_rate":0.0004,#learning rate
	"num_steps":1000000,#the total number of steps to take when training

	#for the loss
	"loss_type":[
		gu.cross_entropy, 
		tf.losses.mean_squared_error, 
		tf.losses.absolute_difference,
		][0],#choose one of these losses.

}
