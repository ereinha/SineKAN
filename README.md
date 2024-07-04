#  SineKAN: Kolmogorov-Arnold Networks Using Sinusoidal Activation Functions

Recent work has established an alternative to traditional multi-layer perceptron neural networks in
the form of Kolmogorov-Arnold Networks (KAN). The general KAN framework uses learnable
activation functions on the edges of the computational graph followed by summation on nodes.
The learnable edge activation functions in the original implementation are basis spline functions
(B-Spline). Here, we present a model in which learnable grids of B-Spline activation functions can
be replaced by grids of re-weighted sine functions. We show that this leads to better or comparable
numerical performance to B-Spline KAN models on the MNIST benchmark, while also providing a
substantial speed increase on the order of 4-9 times

# Google Colab demo

[SineKAN_MNIST_Demo.ipynb](https://github.com/ereinha/SineKAN/blob/main/SineKAN_MNIST_Demo.ipynb)
