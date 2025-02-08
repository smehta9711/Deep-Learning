# 2 Layer Linear Neural Network with SGD

## Overview

This assignment focuses on developing an **age regression model** that predicts the age of an individual from a 48x48 grayscale image. The task involves training a **2-layer linear neural network** using **stochastic gradient descent (SGD)**, implemented solely with NumPy linear algebra operations, avoiding external machine learning libraries. [Dataset](https://drive.google.com/drive/folders/1PUh3xLWbKN0vb35JWhUL8t8rGvagEZcO?usp=sharing)

---

## Problem 2: Training a 2-Layer Neural Network

### Model Description
The model is mathematically represented as:  
\[
\hat{y} = g(x; w, b) = x^T w + b
\]  
where \( w \) represents the weights, \( b \) the bias, and \( \hat{y} \) the predicted age.  
The objective is to minimize the **Mean Squared Error (MSE)** cost function, quantifying the difference between predicted and actual ages.

### Approach
1. **Hyperparameter Optimization**:
   - Fine-tuned hyperparameters: `learning_rate`, `mini_batch_size`, and `number_of_epochs`.
   - Used a validation set created by splitting the training data to ensure robust performance.
2. **Systematic Grid Search**:
   - Tested multiple combinations of hyperparameters using nested loops.
3. **Performance Evaluation**:
   - Evaluated the model on a separate test set after optimization.
   - Reported Training and Testing MSE, along with the cost values from the last 10 iterations of gradient descent.

### Hyperparameter Combinations Tested
- **Learning Rates**: 1e-5, 1e-4, 1e-3  
  (Higher rates, such as 1e-1 and 1.0 caused instability with "NaN" and "inf" values.)
- **Mini-Batch Sizes**: 32, 64, 128  
- **Number of Epochs**: 50, 100, 150  

### Results
The optimal hyperparameters were determined to be:
- **Learning Rate**: \( 1 \times 10^{-4} \)  
- **Mini-Batch Size**: 64  
- **Number of Epochs**: 150  

Performance Metrics:
- **Training MSE**: 0.7645  
- **Testing MSE**: 0.7691  

---

## Problem 3: Challenges in Gradient Descent

This section explores potential challenges in gradient descent optimization, including:

1. **Learning Rate Impact**:
   - Low learning rates ensure stability but lead to slow convergence.
   - High learning rates result in overshooting or divergence.
   - Decaying the learning rate from \( 0.245 \to 0.16 \) enabled smoother convergence.

2. **Directional Oscillations**:
   - Steeper curvature along one axis caused stronger oscillations in that direction.

3. **Function-Specific Behavior**:
   - Investigated convergence for \( f(x) = \frac{2}{3}|x|^{3/2} \) and \( f(x) = x^4 - 4x^2 \) under varying conditions.
   - Highlighted convergence to global minima and oscillatory behavior depending on initialization and learning rate.

### Insights
- Proper learning rate scheduling is crucial for faster and stable convergence.
- Initialization strategies significantly impact optimization stability.
- For highly curved loss functions, gradients along steeper directions cause dominant oscillations.

---

## Repository Contents

- **Code**: Implementations of the 2-layer neural network and gradient descent experiments.
- **Visualizations**: Contour plots and gradient descent trajectories. (In "Report.pdf")
- **Documentation**: Detailed insights on hyperparameter tuning and gradient behavior. (In "Report.pdf")

---


