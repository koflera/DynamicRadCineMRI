import torch
from torchkbnufft import (MriSenseNufft, 
						  AdjMriSenseNufft, 
						  KbNufft, 
						  AdjKbNufft,
						  ToepNufft,
						  ToepSenseNufft)

from torchkbnufft.nufft.toep_functions import calc_toep_kernel
						  

import torch.nn as nn

class Dyn2DRadEncObj(nn.Module):

	"""
	Class for a 2D frame-wise radial operator 

	An instance of the class contains the following methods:
	- apply_E 		... the forward NUFFT operator
	- apply_EH 	... the adjoint operator of E, i.e. E^H
	- apply_EHE	... the composition E^H and E, i.e. E^H o E

	"""

	def __init__(self,im_size, ktraj, dcomp, csmap, norm='ortho'):
		super(Dyn2DRadEncObj, self).__init__()
		"""
		inputs:
		- im_size	... the shape of the object to be imaged
		- ktraj 	... the k-space tractories  (torch tensor)
		- csmap 	... the coil-sensivity maps (torch tensor)
		- dcomp 	... the density compensation (torch tensor)
		
		N.B. the shapes for the tensors are the following:
			
			image   (1,1,2,Nx,Ny,Nt),
			ktraj 	(1,2,Nrad,Nt),
			csmap   (1,Nc,2,Nx,Ny),
			dcomp 	(1,1,1,Nrad,Nt),
			kdata 	(1,Nc,2,Nrad,Nt),
			
		where 
		- Nx,Ny,Nt = im_size
		- Nrad = 2*Ny*n_spokes
		- n_spokes = the number of spokes per 2D frame
		- Nc = the number of receiver coils

		"""
		dtype = torch.float
		
		Nx,Ny,Nt = im_size
		self.im_size = im_size
		self.spokelength = im_size[1]*2
		self.Nrad = ktraj.shape[2]
		self.ktraj_tensor = ktraj
		self.dcomp_tensor = dcomp
		
		#parameters for contstructing the operators
		spokelength=im_size[1]*2
		grid_size = (spokelength,spokelength)

		#single-coil/multi-coil NUFFTs
		if csmap is not None:
			self.NUFFT =  MriSenseNufft(smap=csmap, im_size=im_size[:2], grid_size=grid_size,norm=norm).to(dtype)
			self.AdjNUFFT = AdjMriSenseNufft(smap=csmap, im_size=im_size[:2], grid_size=grid_size,norm=norm).to(dtype)
			self.ToepNUFFT = ToepSenseNufft(csmap).to(dtype)
		else:		
			self.NUFFT = KbNufft(im_size=im_size[:2], grid_size=grid_size,norm=norm).to(dtype)
			self.AdjNUFFT = AdjKbNufft(im_size=im_size[:2], grid_size=grid_size,norm=norm).to(dtype)
			self.ToepNUFFT = ToepNufft().to(dtype)
			
		#calculate Toeplitz kernels
		# for E^{\dagger} \circ E
		self.AdagA_toep_kernel_list = [calc_toep_kernel(self.AdjNUFFT, ktraj[...,kt], 
											 weights=dcomp[...,kt]) for kt in range(Nt) ]
		
		# for E^H \circ E
		self.AHA_toep_kernel_list = [calc_toep_kernel(self.AdjNUFFT, ktraj[...,kt]) for kt in range(Nt) ]

	def apply_A(self, x_tensor):

		#for each time point apply the forward model;
		kdat_list = [self.NUFFT(x_tensor[...,kt],
						   self.ktraj_tensor[...,kt]) for kt in range(self.im_size[2])]
			
		kdat_tensor = torch.stack(kdat_list,dim=-1)
		
		return kdat_tensor

	def apply_AH(self, k_tensor):

		#for each time point apply the adjoint NUFFT-operator;
		xrec_list = [self.AdjNUFFT(k_tensor[...,kt],
							 self.ktraj_tensor[...,kt]) for kt in range(self.im_size[2])] 

		xrec_tensor = torch.stack(xrec_list,dim=-1)
		
		return xrec_tensor
	
	def apply_AHA(self, x):

		#the composition of the operator A^H \circ A
		k = self.apply_A(x)
		x = self.apply_AH(k)

		return x
	
	def apply_dcomp(self, k_tensor, dcomp_tensor):
		
		return self.dcomp_tensor*k_tensor
		
	def apply_Adag(self, k_tensor):
		
		#multiply k-space data with dcomp
		dcomp_k_tensor = self.apply_dcomp(k_tensor, self.dcomp_tensor)

		#apply adjoint
		xrec_tensor = self.apply_AH(dcomp_k_tensor)
		
		return xrec_tensor

	def apply_AdagA(self, x):

		#the composition of the operator A^H \circ A
		k = self.apply_A(x)
		x = self.apply_Adag(k)

		return x

	def apply_AHA_Toeplitz(self, x):

		#for each time point apply Toeplitz NUFFT
		x_toep_list =[self.ToepNUFFT(x[...,kt],
							   self.AHA_toep_kernel_list[kt]) for kt in range(self.im_size[2])]
			
		x_toep_tensor =  torch.stack(x_toep_list,dim=-1)
		
		return x_toep_tensor
	
	def apply_AdagA_Toeplitz(self, x):

		#for each time point apply Toeplitz NUFFT
		x_toep_list =[self.ToepNUFFT(x[...,kt],
							   self.AdagA_toep_kernel_list[kt]) for kt in range(self.im_size[2])]
			
		x_toep_tensor =  torch.stack(x_toep_list,dim=-1)
		
		return x_toep_tensor
	
		
					  
