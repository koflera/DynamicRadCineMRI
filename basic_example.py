"""
Example on how to use the 2d radial encoding operator using pytorch
"""

import torch

from network.nufft_operator import Dyn2DRadEncObj
from helper_funcs.noise_funcs import add_gaussian_noise

import numpy as np

import matplotlib.pyplot as plt

dtype = torch.float

#wheter to use the gpu or not;
use_GPU=1

#load complex-valued ground-truth image of shape (320,320,20)
xf = np.load('toy_data/img_320.npy') 
im_size = xf.shape

#create a torch tensor of shape (1,1,2,320,320,20)
xf_tensor = torch.stack([torch.tensor(xf.real),torch.tensor(xf.imag)],dim=0).unsqueeze(0).unsqueeze(0)

#load k-space trajectories of shape (Nrad,20)
ktraj = np.load('toy_data/ktraj_320.npy') 

#convert to tensor of shape (1,2,Nrad,20)
ktraj_tensor =  torch.stack([torch.tensor(ktraj.real),torch.tensor(ktraj.imag)],dim=0).unsqueeze(0)

#load complex-valued coil-sensitivy maps of shape (12,320,320)
csm = np.load('toy_data/csmap_320.npy')  

#convert to tensor of shape (1,12,2,320,320)
csm_tensor = torch.stack([torch.tensor(csm.real),torch.tensor(csm.imag)],dim=1).unsqueeze(0) 

#load density compensation function
dcomp = np.load('toy_data/dcomp_320.npy')  #shape (Nrad,20)

dcomp_tensor = torch.tensor(dcomp).unsqueeze(0).unsqueeze(0).unsqueeze(0) 

if use_GPU:
	xf_tensor = xf_tensor.to('cuda')
	ktraj_tensor = ktraj_tensor.to('cuda')
	csm_tensor = csm_tensor.to('cuda')
	dcomp_tensor = dcomp_tensor.to('cuda')
	
#create encoding operator object
EncObj = Dyn2DRadEncObj(im_size,ktraj_tensor,dcomp_tensor,csm_tensor,norm='ortho').cuda()

#forward; add noise as well
k_tensor = EncObj.apply_A(xf_tensor)
k_tensor = add_gaussian_noise(k_tensor,sigma=0.025)
	
#undersampled reco
xu_tensor = EncObj.apply_Adag(k_tensor)

if use_GPU:
	xu_tensor = xu_tensor.cpu()

#convert to numpy
xu = xu_tensor.squeeze(0).squeeze(0).numpy()
xu = xu[0,...] + 1j*xu[1,...]

#save figure
fig,ax=plt.subplots(1,3,figsize=(3*5,5*1))
ax[0].imshow(np.abs(xf)[:,:,0],cmap=plt.cm.Greys_r,clim=[0,1000])
ax[0].set_title('ground-truth')
ax[1].imshow(np.abs(xu)[:,:,0],cmap=plt.cm.Greys_r,clim=[0,1000])
ax[1].set_title('undersampled reco')
ax[2].imshow(np.abs(xu-xf)[:,:,0],cmap=plt.cm.viridis,clim=[0,1000])
ax[2].set_title('error')
plt.tight_layout()
fig.savefig('results/basic_example.pdf')
