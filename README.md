# COMP 4107 Hopfield_SOM_PCA

Carolyne Pelletier

Akhil Dalal

Question 1: Hopfield Network on MNIST Data using images of ’1’ and ’5’ only

	To run: Python3 Question1.py
	
Question 2: Self-Organizing Maps (SOM) as a Substitute for K-means on Normalized MNIST Data using images of ’1’ and ’5’ only

	To run: Python3 Question2.py
  
	- K-Means will run first, generating 5 .png files in root directory depicting k-means with 2-6 clusters 
	- K-Means was reduced using PCA rather than SVD 
	- SOM will run second, generating 2 .png files in root directory depicting 15x15 SOM pre-training (random initial weights) and 15x15 SOM post-training (organized clusters) with a learning rate of 0.1

Question 3: Principal Component Analysis on Scikit-Learn Face Data

	To run: Python3 Question3.py
	
	-Tested with tensor flow version 1.4.0 on macOS
