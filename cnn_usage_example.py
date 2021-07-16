"""
Example on how to apply a pre-trained reconstruction network.
Note that the network consists of an alternating sequence of CNN-block
and CG-block. This script shows that, as also explained in the paper,
the length of the network can be varied at test time by still being able
to observe some increase of performance, although the netork was trained
with only one CNN- and CG-block, i.e. with length one.
This suggests that finding a compromise between the depth of the CNN-block
and the overall length of the network might be not necessary and therefore,
choosing more expressive CNN-blocks is a viable option even for large-scale
problems.
"""

import torch

import sys
sys.path.append('network/')

from network.nufft_operator import Dyn2DRadEncObj
from network.reconstruction_network import NUFFTCascade
from network.xtyt_fft_unet import XTYTFFTCNN

from helper_funcs.noise_funcs import add_gaussian_noise

import numpy as np

import matplotlib.pyplot as plt
plt.ioff()

from skimage.measure import compare_nrmse

#datatype
dtype = torch.float

#wheter to use the gpu or not;
use_GPU=1

############
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

#define the CNN-block and thus the model; 
#available: E3C2K4, E3C2K8, E3C2K16;
n_enc_stages = 3
n_convs_per_stage = 2
n_filters = 16
CNN = XTYTFFTCNN(n_ch=2,
				 n_enc_stages=n_enc_stages,
				 n_convs_per_stage=n_convs_per_stage,
				 n_filters=n_filters)

#initialize reconstruction network
reconstruction_network = NUFFTCascade(EncObj,
									  CNN,
									  learn_lambda=False,
									  use_precon=True,
									  mode='fine_tuning').cuda()

model_folder = 'pre_trained_models/'
model_id='E{}C{}K{}'.format(n_enc_stages, n_convs_per_stage, n_filters)

reconstruction_network.load_state_dict(torch.load(
	model_folder+'model_{}.pt'.format(model_id)
	))
#forward; add noise as well
k_tensor = EncObj.apply_A(xf_tensor)
k_tensor = add_gaussian_noise(k_tensor,sigma=0.06)
	
#undersampled reco
xu_tensor = EncObj.apply_Adag(k_tensor)

#list of different parameters for nu and npcg as used in the paper
nu_list = [1,1,2,4,8,12]
npcg_list = [0,8,4,4,4,4]

#initialize dictionary which contains recos for different hyper-parameters
D_cnn_recos = {}
n_tests=6
for nu, npcg in list(zip(nu_list,npcg_list))[:n_tests]:
	
	reconstruction_network.nu = nu
	reconstruction_network.npcg = npcg
	
	#apply CNN-block + CG-block
	with torch.no_grad():
		xcnn_reg = reconstruction_network(xu_tensor.squeeze(0))
		
	if use_GPU:
		xcnn_reg = xcnn_reg.cpu()
		
	xcnn_reg = xcnn_reg.squeeze(0).squeeze(0).numpy()
	xcnn_reg = xcnn_reg[0,...] + 1j*xcnn_reg[1,...]
	D_cnn_recos['xcnn_nu{}_npcg{}'.format(nu,npcg)] = xcnn_reg
		
if use_GPU:
	xu_tensor = xu_tensor.cpu()

xu = xu_tensor.squeeze(0).squeeze(0).cpu().numpy()
xu = xu[0,...] + 1j*xu[1,...]

#create and save figure
n_tests = len(D_cnn_recos.keys())
n_subfigs = n_tests+2

fig,ax=plt.subplots(2,n_subfigs,figsize=(n_subfigs*8,8*2.5))
clim = [0,1000]
cutoff=80
font_size=32
arrs_list = [xu] + [D_cnn_recos['xcnn_nu{}_npcg{}'.format(nu,npcg)] for nu, npcg in list(zip(nu_list,npcg_list))] + [xf]
errs_list = [3*(arr - xf) for arr in arrs_list]
names_list = ['undersampled reco'] + ['xcnn\n nu{},npcg{}'.format(nu,npcg) 
									  for nu, npcg in list(zip(nu_list,npcg_list)) ] + ['ground-truth'] 

for k in range(n_subfigs):
	arr = arrs_list[k][cutoff:320-cutoff,cutoff:320-cutoff,10]
	err = errs_list[k][cutoff:320-cutoff,cutoff:320-cutoff,10]
	name = names_list[k]
	
	ax[0,k].imshow(np.abs(arr),cmap=plt.cm.Greys_r,clim=clim)
	ax[1,k].imshow(np.abs(err),cmap=plt.cm.viridis,clim=clim)
	
	#compute NRMSE
	nrmse = np.mean([compare_nrmse(np.abs(xf[cutoff:320-cutoff,cutoff:320-cutoff,10]), np.abs(arr))])
	nrmse = np.round(nrmse,decimals=4)
	print('nrmse = {}'.format(nrmse))
	ax[0,k].set_title(name,fontsize=font_size)
	ax[1,k].set_title('NRMSE={}'.format(nrmse),fontsize=font_size)
	
for kx in range(2):
	for ky in range(n_subfigs):
		ax[kx,ky].set_xticks([])
		ax[kx,ky].set_yticks([])	
plt.tight_layout()
fig.savefig('results/nu_npcg_variation_{}.pdf'.format(model_id))
