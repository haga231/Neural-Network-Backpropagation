# Neural Network with Backpropagation

## 👩‍🎓 Student Information
- **Name:** Hagar Ashraf El_Bialy
- **ID:** 4231064

## 📝 Description
This project implements a neural network with one hidden layer using the tanh activation function. It demonstrates both feed forward propagation and back propagation algorithms from scratch using NumPy.

## 🏗️ Network Architecture
- **Input Layer:** 2 neurons
- **Hidden Layer:** 2 neurons
- **Output Layer:** 1 neuron
- **Activation Function:** tanh (hyperbolic tangent)

## ⚙️ Parameters
- **Initial Weights:** Random values in range [-0.5, 0.5]
- **Biases:** b1 = 0.5 (hidden layer), b2 = 0.7 (output layer)
- **Input Data:** X = [0.5, 0.3]
- **Target Output:** y_true = 0.8
- **Learning Rate:** α = 0.1

## 🎯 Objectives Achieved
1. ✅ Feed forward propagation to calculate network output
2. ✅ Back propagation to compute gradients
3. ✅ Weight update using gradient descent
4. ✅ Verify that the network learns (error decreases)

## 🚀 How to Run

### Prerequisites
```bash
pip install numpy
