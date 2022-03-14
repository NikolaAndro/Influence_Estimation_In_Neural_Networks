# Memorization & Influence Estimation In Neural Networks
Discovering the Long Tail via Influence Estimation.

The MNIST-1D dataset is obtained using the code in https://colab.research.google.com/github/greydanus/mnist1d/blob/master/building_mnist1d.ipynb (Links to an external site.) , it is assigned to a variable named data. 

Using the estimators described in Algorithm 1 of https://arxiv.org/pdf/2008.03703.pdf (Links to an external site.) for the models ConvBase and MLPBase, I will compute memorization estimates for each training data point (influence of each training point on each testing point is commented out in the code, but is still there). 

Memorization values are plotted as histograms for each model type. 

Then, I am constructing a different MNIST 1D dataset by changing one of the dataset arguments such that the number of high memorization examples increases. 

Finally, new memorization values are computed and saved as histograms. 

