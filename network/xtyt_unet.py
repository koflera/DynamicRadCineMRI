import torch.nn as nn

import numpy as np

from unet import UNet

class XTYTUNet(nn.Module):
	
	""" 
	Create a XT,YT U-Net
	
	the network is used to process a 2D cine MR image of shape
	(1,2,Nx,Ny,Nt)
	
	the CNN first "rotates" the sample to the xt- and the yt-view,
	then applies a U-Net on the spatio-temporal slices and 
	then re-assembles to cine MR image from the processed slices.
	
	N.B. 
	i) as a default, the CNN used for the xt-view and the yt-view is the same
	since radial-undersampling artefacts have a "noise-like" structure.
	For different sampling patterns, one could set weight_sharing to False
	
	ii) Note that wheter to use the residual connection or not, is decided in
			the class XTYTFFTCNN
	
	"""
	def __init__(self,n_ch_in=2,n_ch_out=2, n_enc_stages=3, n_convs_per_stage=4,n_filters=64,weight_sharing=True):
		super(XTYTUNet, self).__init__()

		self.n_ch_in = n_ch_in
		self.n_ch_out = n_ch_out
		self.n_filters = n_filters
		self.n_convs_per_stage = n_convs_per_stage
		self.weight_sharing = weight_sharing
		self.n_enc_stages=n_enc_stages
		
		#dimensionality of the U-Net; this is alwys 2 for the XT,YT-Net
		dim=2
		
		#if weight sharing is applied for the xt- and the yt-CNN,
		#might me beneficial for Cartesian sampling trajectories, for example;
		if weight_sharing:

			self.conv_xt_yt = UNet(dim,n_ch_in=n_ch_in,n_ch_out=n_ch_out,n_enc_stages=n_enc_stages,n_convs_per_stage=n_convs_per_stage,
						  n_filters=n_filters)
			
		else:
			self.conv_xt = UNet(dim,n_ch_in=n_ch_in,n_ch_out=n_ch_out,n_enc_stages=n_enc_stages,n_convs_per_stage=n_convs_per_stage,
						  n_filters=n_filters)
			self.conv_yt = UNet(dim,n_ch_in=n_ch_in,n_ch_out=n_ch_out,n_enc_stages=n_enc_stages,n_convs_per_stage=n_convs_per_stage,
						  n_filters=n_filters)
			
		self.reshape_op_xyt2xt_yt = XYT2XT_YT()
		self.reshape_op_xt_yt2xyt = XT_YT2XYT()

	def forward(self, x):
		
		#get the number of sampels used; needed for re-assembling operation
		# x has the shape (mb,2,nx,ny,nt)
		mb = x.shape[0]

		#input is 5d -> output is 4d
		x_xt = self.reshape_op_xyt2xt_yt(x,'xt')
		x_yt = self.reshape_op_xyt2xt_yt(x,'yt')
				
		#input is 4d
		if self.weight_sharing:
			x_xt_conv = self.conv_xt_yt(x_xt)	
			x_yt_conv = self.conv_xt_yt(x_yt)	
		else:
			x_xt_conv = self.conv_xt(x_xt)
			x_yt_conv = self.conv_yt(x_yt)	

		#input is 4d -> output is 5d
		x_xt_r = self.reshape_op_xt_yt2xyt(x_xt_conv,'xt',mb)
		x_yt_r = self.reshape_op_xt_yt2xyt(x_yt_conv,'yt',mb)

		#5d tensor
		x = 0.5*(x_xt_r + x_yt_r)

		return x
	
	
class XYT2XT_YT(nn.Module):
	""" 
	Class needed for the reshaping operator:
	Given x with shape (mb,2,Nx,Ny,Nt), x is reshped to have
	either shape (mb*Nx,2,Ny,Nt) for the yt-domain or 
	the shape (mb*Ny,2,Nx,Nt) for the xt-domain
	"""
	
	def __init__(self):
		super(XYT2XT_YT, self).__init__()

	def forward(self, x, reshape_type):

		return xyt2xt_yt(x,reshape_type)



def xyt2xt_yt(x,reshape_type):

	#x has shape (mb,2,nx,ny,nt)		
	mb,nch,nx,ny,nt = x.shape

	if reshape_type=='xt':
		x = x.permute(0,2,1,3,4).view(mb*nx, nch, ny, nt)

	elif reshape_type =='yt':
		x = x.permute(0,3,1,2,4).view(mb*ny, nch, nx, nt)
	
	return x 


class XT_YT2XYT(nn.Module):
	""" 
	Class needed for the reassembling the cine MR image to its original shape:
	reverses the operation XYT2XT_YT,
	note that the mini-batch size is needed
	"""
	
	def __init__(self):
		super(XT_YT2XYT, self).__init__()

	def forward(self, x, reshape_type,mb):
		
		return xt_yt2xyt(x, reshape_type,mb)


def xt_yt2xyt(x,reshape_type,mb):

	if reshape_type =='xt':

		_,nch,ny,nt=x.shape
		nx = np.int(x.shape[0]/mb)

		x = x.view(mb,nx,nch,ny,nt).permute(0,2,1,3,4)
	
	elif reshape_type=='yt':

		_,nch,nx,nt=x.shape
		ny = np.int(x.shape[0]/mb)

		x = x.view(mb,ny,nch,nx,nt).permute(0,2,3,1,4)
	
	return x 