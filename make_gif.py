import imageio as iio 
import numpy as np 
import os
#creates the give from prediction images.
image_path = "predictions"
use_images = True#whether to use the images or create from the numpy saves
def load_all_images(path):
	image_names = {int(i.split("_")[1][:-4]):os.path.join(path, i) for i in os.listdir(path) if i.endswith(".jpg")}
	image_index = list(image_names.keys())
	image_index.sort()
	image_names = [image_names[i] for i in image_index]
	images = [iio.imread(i) for i in image_names]
	return images

if use_images:
	images = load_all_images(image_path)
else:
	print("not implemented yet")
	exit()

iio.mimsave("%s.gif"%image_path, images, duration=1/64)