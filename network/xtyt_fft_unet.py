import torch
import torch.nn as nn

from xtyt_unet import XTYTUNet

class XTYTFFTCNN(nn.Module):

	"""
	Implementation of a simple CNN consisting of a UNet which is applied 
	in the Fourier-transformed spatio-temporal domain. 

	input parameters for the construction of the CNN:
	- n_ch			 - number of input channels, default is 2 for complex-numbers
	- n_enc_stages 	 - number of encoding stages of the U-net
	- n_convs 	 	 - number f conv layers per stage
	- n_filters		 - number of filters used for the first convolutional layer
	- weight_sahring - wheter to apply weight sharing of the two blocks of the conv layers or not;
	- res_connection - wheter to use a residual connection or not;
	
	"""

	def __init__(self,n_ch=2, n_enc_stages=3, n_convs_per_stage=4, n_filters=32, weight_sharing=True, res_connection=True):
		super(XTYTFFTCNN, self).__init__()
		
		self.weight_sharing = weight_sharing
		self.n_enc_stages = n_enc_stages
		self.n_filters = n_filters
		self.n_convs_per_stage = n_convs_per_stage
		self.res_connection = res_connection

		#the CNN block c_{\Theta}
		self.cnn = XTYTUNet(n_ch_in=2, 
						  n_ch_out=2, 
						  n_enc_stages=n_enc_stages,
						  n_convs_per_stage=n_convs_per_stage,
						  n_filters=n_filters,
						  weight_sharing=weight_sharing)
			
	def forward(self, x):
		
		mb,nch,Nx,Ny,Nt = x.shape
		
		if self.res_connection:
			xu = x.clone()
			xmu = torch.stack(Nt*[torch.mean(xu,dim=-1)],dim=-1)
			x=x-xmu
			
		#apply temporal FFT
		x = x.permute(0,2,3,4,1)
		x = torch_fftshift(torch.fft(torch_ifftshift(x, dims=[-2]),  1, normalized=True), dims=[-2])
		x = x.permute(0,4,1,2,3)
		

		#CNN opearting on temporal transformed xt,yt-domain
		x = self.cnn(x)
		
		#apply temporal IFFT
		x = x.permute(0,2,3,4,1)
		x = torch_fftshift(torch.ifft(torch_ifftshift(x, dims=[-2]),  1, normalized=True), dims=[-2])
		x = x.permute(0,4,1,2,3)
		
		#residual connection
		if self.res_connection:
			 x=x+xmu
			  
		return x
	
	
def fftshift(x, dims, offset=1):
    
    x_shape = x.shape
    ndim = len(x_shape)
    dims = [(ndim + dim) % ndim for dim in dims]
	
    for dim in dims:

        if x_shape[dim] == 1:
            continue
        n = x_shape[dim]
        half_n = (n + offset)//2
        curr_slice = [ slice(0, half_n) if i == dim else slice(x_shape[i]) for i in range(ndim) ]
        curr_slice_2 = [ slice(half_n, x_shape[i]) if i == dim else slice(x_shape[i]) for i in range(ndim) ]
        x = torch.cat([x[curr_slice_2], x[curr_slice]], dim=dim)
    return x


def torch_fftshift(x, dims):
    return fftshift(x, dims, offset=1)


def torch_ifftshift(x, dims):
    return fftshift(x, dims, offset=0)