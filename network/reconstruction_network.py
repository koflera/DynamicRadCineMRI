import torch
import torch.nn as nn

import numpy as np

class NUFFTCascade(nn.Module):

	def __init__(self, EncObj, CNN, nu=1,npcg=4,mode='pre_training', use_precon=True, learn_lambda=True):

		"""
		A CNN which consists of alternating blocks of CNNs and CGs; 
		- the forward model is the NUFFT operator;
		- the CNN can be any CNN suitable for processing a a 2D cine MR image 
		
		Parameters:
			
			- EncObj: 	- the econding operator with the forward E , adjoint E^h and composite operator H:= E^H \circ E
			- CNN: 	 	- the CNN-block to be used
			- nu: 	 	- the number of alternations between CG- and CNN modules
			- npcg: 	- te number of  CG iterations in the CG-module;
			- mode: 	- defines whether we a re in the pretraining (only CNN) or inthe fine-tuning mode (CNN+CG) ;
			- use_precon: - defines whether to pre-condition the problem by using the density compensation function
			- lambda_reg - the regularization parameter; if None, it is learned during training;
								
		"""
		
		assert mode in ['pre_training','fine_tuning','testing'], \
			"mode has to be one of 'pre_training', 'fine_tuning' or 'testing"
		
		super(NUFFTCascade, self).__init__()
		
		self.cnn =CNN
		self.EncObj  = EncObj
		self.nu = nu
		self.npcg = npcg
		self.mode=mode
		self.use_precon = use_precon
		
		# activation function which is used to constrain the 
		# learned regularization parameter to be strictly positive
		beta=1.
		self.Softplus = nn.Softplus(beta) 

		if learn_lambda:
			requires_grad=True 
		else:
			requires_grad=False 
			
		lambda_init = np.log(np.exp(1)-1.)/1.
		self.lambda_reg = nn.Parameter(torch.tensor(lambda_init*torch.ones(1),dtype=torch.float),
								requires_grad=requires_grad)
			
	def HOperator(self,x):
		
		#the operator H = A^H \circ A + \lambda_Reg * \Id
		if self.mode in ['pre_training','fine_tuning']:
			if self.use_precon:
				x = self.EncObj.apply_AdagA_Toeplitz(x) + self.Softplus(self.lambda_reg)*x
			else:
				x = self.EncObj.apply_AHA_Toeplitz(x) + self.Softplus(self.lambda_reg)*x
				
		elif self.mode in ['testing']:
			if self.use_precon:
				x = self.EncObj.apply_AdagA(x) + self.Softplus(self.lambda_reg)*x
			else:
				x = self.EncObj.apply_AHA(x) + self.Softplus(self.lambda_reg)*x
				
			 
		return x
	
	def ConjGrad(self, H, x, b, niter=4):
		
		#x is the starting value, b the rhs;
		r = H(x)
		r = b-r
		
		#initialize p
		p = r.clone()
		
		#old squared norm of residual
		sqnorm_r_old = torch.dot(r.flatten(),r.flatten())
		
	
		for kiter in range(niter):
		
			#calculate Hp;
			d = H(p);
	
			#calculate step size alpha;
			inner_p_d = torch.dot(p.flatten(),d.flatten())
			alpha = sqnorm_r_old / inner_p_d
	
			#perform step and calculate new residual;
			x = torch.add(x,p,alpha= alpha.item())
			r = torch.add(r,d,alpha= -alpha.item())
			
			#new residual norm
			sqnorm_r_new = torch.dot(r.flatten(),r.flatten())
			
			#calculate beta and update the norm;
			beta = sqnorm_r_new / sqnorm_r_old
			sqnorm_r_old = sqnorm_r_new
	
			p = torch.add(r,p,alpha=beta.item())

		return x
		
	def forward(self,x):
		
		if self.mode =='pre_training':
			
			# apply the CNN
			x = self.cnn(x)
			
		elif self.mode in['fine_tuning','testing']:
			
			# initial NUFFT reconstruction; 
			# shape (1,2,Nx,Ny,Nt)
			xu = x.clone()
			
			#unsqueeze to (1,1,2,Nx,Ny,Nt)
			xu = xu.unsqueeze(0)
			
			for k in range(self.nu):
				
				# apply the CNN; 
				# shape (1,2,Nx,Ny,Nt)
				x = self.cnn(x) 
				
				# perform CG
				if self.npcg!=0:
					
					# shape (1,1,2,Nx,Ny,Nt)
					x = self.ConjGrad(self.HOperator, x.unsqueeze(0), xu+self.Softplus(self.lambda_reg)*x.unsqueeze(0), niter=self.npcg)
					
					# reduce shape to (1,2,Nx,Ny,Nt)
					x = x.squeeze(0)	
			
		return x
