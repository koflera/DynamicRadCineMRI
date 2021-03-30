import torch
import numpy as np

def add_gaussian_noise(kdata,sigma=0.02):
	
	"""
	function for adding normally-distributed noise to the measured
	k-space data.
													
	"""
	np.random.seed(0)
	
	#the deivice on which the k-space data is located;
	device = kdata.device
	sigma= torch.tensor(sigma).to(device)
	
	mb, Nc, n_ch, Nrad, Nt = kdata.shape 
		
	#center the data and add normally distributed noise
	for kc in range(Nc):
		for kt in range(Nt):
			mu, std = torch.mean(kdata[:,kc,:,:,kt]), torch.std(kdata[:,kc,:,:,kt])
			
			kdata[:,kc,:,:,kt]-=mu
			kdata[:,kc,:,:,kt]/=std
			
			torch.manual_seed(0)
			noise = sigma*torch.randn(kdata[:,kc,:,:,kt].shape).to(device)
			kdata[:,kc,:,:,kt]+= noise
			
			kdata[:,kc,:,:,kt] = std*kdata[:,kc,:,:,kt] + mu

	return kdata