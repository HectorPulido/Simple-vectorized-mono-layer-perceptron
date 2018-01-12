# Vectorized monolayer perceptron

This is a simple perceptron made with [Simple Linear Algebra for C#](https://github.com/HectorPulido/Simple_Linear_Algebra) , is a neural network that can calcule Xor Xnor And Or via Stochastic gradient descent backpropagation with Sigmoid and Relu as Activation function.

There is a lot to improve, like csv read, gpu implementation, regularization, but is functional.
 
## How use it
Just go to the project and open Program.cs and run it, you can change the dataset changing X and Y variables

## How use Relu
0. Change all sigmoid function, for relu function
1. a3 must have no Nonlinear function Matrix a3 = z3;
2. because of that Delta3 has not derivated Matrix Delta3 = a3Error * 1;
3. The learning rate must be smaller, like 0.001 

## Where can i learn more
- On my Youtube channel (spanish) are a lot of information about Machine learning and Neural networks
- https://www.youtube.com/channel/UCS_iMeH0P0nsIDPvBaJckOw
- You can also look at the generalized Example of This 
- https://github.com/HectorPulido/Vectorized-multilayer-neural-network
- Or Look at a Non Vectorized multilayer perceptronExample
- https://github.com/HectorPulido/Multi-layer-perceptron

## Patreon
Please consider Support on Patreon
https://www.patreon.com/HectorPulido

