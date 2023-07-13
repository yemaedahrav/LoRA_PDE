# Weight Decomposition for PINNs

Exploring Matrix Decomposition to remove weight spaces fluctuations and dissimilarities in context of PDEs


## Experiment Setup and Structure

There are 80 boundary points along each side of the domain (each edge) of the rectangle. We have 2500 colocation points sampled using the latin hypercube sampling in the cartesian rectangular domain. The colocation points are used for the physics loss and the dirichlet boundary condition points are used for computing the data loss. 

Based on experimentations results the pretraining and finetuning learning rate schedules are as follows. For trainign with random initializations we train for 50k epochs with the starting learning rate as 0.01 with a decay of 0.1 for every 8k steps. For the finetuning from a particular PINN the step size is set to be 5k and the model is trained for a total of 30k epochs.
The optimizer used is Adam. L-BFGS is another popular choice for PINN based approaches.