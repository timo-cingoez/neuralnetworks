# Michael A. Nielsen, "Neural Networks and Deep Learning" - Exercises

### Sigmoid neurons simulating perceptrons, part I
"""
* Suppose we take all the weights and biases in a network of perceptrons, 
* and multiply them by a positive constant, c>0.
* Show that the behaviour of the network doesn't change.
"""

def perceptron(inputs, weights, bias, constant=1):
    activation_sum = 0
    for i in range(len(inputs)):
        activation_sum += (weights[i]*constant) * (inputs[i]*constant)
    activation_sum += bias
    activation_res = 1 if activation_sum > 0 else 0
    return activation_sum, activation_res

print('perceptron', perceptron([2, 2], [-2, -2], 3))
print('perceptron', perceptron([2, 2], [-2, -2], 3, 7))


### Sigmoid neurons simulating perceptrons, part II
"""
* Suppose we have the same setup as the last problem - a network of perceptrons.
* Suppose also that the overall input to the network of perceptrons has been chosen.
* We won't need the actual input value, we just need the input to have been fixed.
* Suppose the weights and biases are such that w⋅x+b≠0 for the input x
* to any particular perceptron in the network.
* Now replace all the perceptrons in the network by sigmoid neurons,
* and multiply the weights and biases by a positive constant c>0.
* Show that in the limit as c→∞ the behaviour of this network of sigmoid neurons
* is exactly the same as the network of perceptrons. 
* How can this fail when w⋅x+b=0 for one of the perceptrons?
"""

import math

def sigmoid_neuron(inputs, weights, bias, constant=1):
    activation_sum = 0
    for i in range(len(inputs)):
        activation_sum += (weights[i]*constant) * (inputs[i]*constant)
    activation_sum += bias
    activation_res = sigmoid(activation_sum)
    perceptron_res = 1 if activation_res > 0.5 else 0 
    return activation_sum, activation_res, perceptron_res

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

print('sigmoid neuron', sigmoid_neuron([2, 2], [-2, -2], 3))
print('sigmoid neuron', sigmoid_neuron([2, 2], [-2, -2], 3, 7))
