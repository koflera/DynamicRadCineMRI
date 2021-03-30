# DynamicRadCineMRI
Implementation of an iterative network for 2D radial cine MRI reconstruction with multiple receiver coils

Here, we provide code for our paper

"An End-To-End-Trainable Iterative Network Architecture for Accelerated Radial Multi-Coil 2D Cine MR Image Reconstruction"

by A. Kofler, M. Haltmeier, T. Schaeffter, C. Kolbitsch.

The code contains the following major blocks which are used for the construction of the iterative network.

- the proposed CNN-block
- the encoding operator; contains the forward model A, the adjoint A^H, the density-compensated adjoint operator A^# = A^H \circ W, where W denotes a diagonal operator containing the values of the density-compensation function as well as the compositions A^H \circ A and A^# \circ A.
- the implementation of the operator H = A^H \circ A + \lambda \Id and H = A^# \circ A + \lambda \Id, respectively
- an implementation of a CG module to solve the system Hx=b


Further, we provide an example of toy-data which can be used to get familiar with the code. 
- img_320.npy:    -  the ground-truth image x
- ktraj_320.npy:  -  a set of k-space trajectories chosen according to the golden-angle method
- dcomp_320.npy:  -  the values of the density-compensation function to be used in the operator A^#
- csmap_320.npy:  -  coil-sensitvity maps for 12 receiver coils to be used in the operators

The image in the file img_320.npy was borrowed from http://www.cse.yorku.ca/~mridataset/.

>> N.B. For using this code, you have to use PyTorch version 1.6.0 as well as TorchKbNufft version 0.3.4 (https://github.com/mmuckley/torchkbnufft)

>> Further, note that in this implementation, the forward and adjoint NUFFT-operators are defined beforehand. Using this version of TorchKbNufft, this is required for using the classes MriSenseNufft/AdjMriSenseNufft. This means that, when training, only one set of k-space trajectories, density-compensation function and coil-sensitivity maps is used. This also means that the mini-batch size used for fine-tuning and testing has to be mb=1. However, at test time, you can of course set the csm, dcomp and ktraj according to the considered patient-specific setting.
This can in principle be circumvented by upgrading to the newest TorchKbNufft version, where csm can be used when calling the operators. In future, we might upgrade the code to be compatible with pytorch >1.6.0 and TorchKbNufft >0.3.4.

If you find the code useful or you use it in your work, please cite our work:

@article{kofler2021end,
  title={An End-To-End-Trainable Iterative Network Architecture for Accelerated Radial Multi-Coil 2D Cine {MR} Image Reconstruction},
  author={Kofler, Andreas and Haltmeier, Markus and Schaeffter, Tobias and Kolbitsch, Christoph},
  journal={Medical Physics},
  year={2021},
  DOI={10.1002/mp.14809},
  publisher={Wiley Online Library}
}

The pre-print of our paper can also be found at
https://arxiv.org/abs/2102.00783
