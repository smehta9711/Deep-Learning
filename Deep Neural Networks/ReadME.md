# **Multi-Layer Feed-Forward Neural Network for Fashion MNIST Classification**

This project develops a **multi-layer feed-forward neural network** to classify images of fashion items from the [Fashion MNIST Dataset](https://github.com/zalandoresearch/fashion-mnist). The dataset consists of **10 different classes** representing various clothing categories. The network processes **28x28 pixel grayscale images**, which are flattened into **784-dimensional vectors**, and outputs a **probability distribution** over the classes.

The key focus of this project is implementing the **backpropagation algorithm from scratch** to efficiently compute gradients for training the network.

---

## **Neural Network Architecture**
The architecture of the neural network follows these equations:

1. **Forward Pass**:
\[
  z^{(1)} = W^{(1)}x + b^{(1)} \quad \quad h^{(1)} = \text{ReLU}(z^{(1)})
\]
\[
  z^{(2)} = W^{(2)}h^{(1)} + b^{(2)} \quad \dots \quad z^{(l)} = W^{(l)}h^{(l-1)} + b^{(l)}
\]
\[
  \hat{y} = \text{softmax}(z^{(l)})
\]

2. **Cross-Entropy Loss**:
The unregularized cross-entropy cost function is defined as:
\[
  f_{CE}(W^{(1)}, b^{(1)}, \dots, W^{(l)}, b^{(l)}) = -\frac{1}{n} \sum_{i=1}^n \sum_{k=1}^{10} y_k^{(i)} \log \hat{y}_k^{(i)}
\]

Where:
- **n** is the number of training examples,
- **y\(i\)** represents the true labels,
- **\hat{y}\(i\)** are the predicted probabilities.

---

## **Key Components**

### 1. **Backpropagation Implementation**
A primary focus of this project is implementing the **backpropagation algorithm from scratch**. Backpropagation efficiently calculates the gradients of the loss function with respect to the network's weights and biases. These gradients are used to update the network parameters during training.

---

### 2. **Hyperparameter Tuning**
We explore various hyperparameters and architectural choices to optimize network performance, including:
- **Number of hidden layers**: 3, 4, or 5
- **Number of units per hidden layer**: 30, 40, or 50
- **Learning rate**: Ranging from 0.001 to 0.5
- **Mini-batch size**: 16, 32, 64, 128, or 256
- **Number of epochs**
- **L2 Regularization Strength**: Regularization applied to weight matrices
- **Learning rate decay**: Decay frequency and rate
- **Data augmentation techniques**: To enhance generalization

---

### 3. **Gradient Checking**
To ensure the correctness of gradient computations, we implement **gradient checking** using the provided `check_grad` method. This involves:
- Numerically estimating gradients using finite differences
- Comparing these estimates with analytically computed gradients

**Gradient verification** results are included for a network with:
- **3 hidden layers**
- **64 neurons** per hidden layer

---

## **Tasks**

1. **Stochastic Gradient Descent (SGD)**
   - Implement **SGD** to train the multi-layer network.
   - Ensure that backpropagation supports any number of hidden layers.

2. **Gradient Verification**
   - Include results of **gradient checking** for the specified network configuration.

3. **Performance Tracking**
   - Report the following metrics:
     - **Test Accuracy**
     - **Unregularized Cross-Entropy Loss**
   - Aim for a **test accuracy of at least 88%**.

4. **Weight Visualization**
   - Visualize the **first layer of weights** by reshaping them into **28x28 matrices**.
   - Present these weight matrices as images in a grid.

---
