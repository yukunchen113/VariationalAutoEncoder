# Variational Auto Encoder
## About this Project
Here is a simple implementation of a VAE using tensorflow.

The parameters to be tuned can be accessed in params.py. Analysis below used these parameters.

The purpose of this repository is to learn and test an understanding of VAES. Please see the _VAE background_ section, this will get us an understanding of VAEs, which we can then test. See the _Analysis_ section for an analysis of the VAE.

## Requirements
Please pull my utils repo, and add it to your python path. The functions there are used.

The model is contained in main.py, the parameters are in params.py and the predictions folder contains saved images during training.

## VAE background
### Resources
These are great resources for VAEs:
- Original Paper: [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)
- Explanation: [Tutorial on Variational Autoencoders](https://arxiv.org/abs/1606.05908)

### Theory
#### What is a Latent space?
A


### Implementation Tricks


## Analysis:
- cross entropy vs MSE vs KLD annealing
	- CE seems to counter KLD more so than MSE
- tuning the architecture (layer sizes, amounts)
- Analysis of the latent space:
	- tuning latent representation size
	- as the representation changes
	- as the representation increases in size
	- plot of space with umap/tsne
	- Transformation between two reconstructions
	- analysis of effect on latent space per sample
		- as a measure of distance.
	- VAE space vs AE space
- changing datasets

## Going Beyond 
Add in noise as part of the latent representation,
- Use a very small latent representation which is trained, rest is noise.

## Future Work
- add tensorboard