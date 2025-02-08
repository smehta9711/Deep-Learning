# **2-Layer Softmax Neural Network for Fashion MNIST Classification**

This project implements a **2-layer softmax neural network** to classify images of fashion items from the [Fashion MNIST Dataset](https://drive.google.com/drive/folders/11jvHCv12x5uOpKgnda2F8DH4S4Hiigv-?usp=sharing). The network takes **28x28 pixel grayscale images** (flattened into **784-dimensional vectors**) and outputs a vector of **10 probabilities** corresponding to different fashion categories, such as shoes, t-shirts, and dresses.

The objective is to **minimize the cross-entropy loss function**:

\[
    f_{CE}(\mathbf{w}^{(1)}, \dots, \mathbf{w}^{(10)}, \mathbf{b}^{(1)}, \dots, \mathbf{b}^{(10)}) = -\frac{1}{n} \sum_{i=1}^n \sum_{k=1}^{10} y_k^{(i)} \log \hat{y}_k^{(i)} + \frac{\alpha}{2} \sum_{k=1}^c \mathbf{w}^{(k)} \mathbf{w}^{(k)\top}
\]

Where:
- **n** is the number of training examples,
- **y(i)** represents the true labels,
- **\hat{y}(i)** are the predicted probabilities,
- **\alpha** is the regularization constant.

Regularization is applied **only to the weights**, while biases remain unregularized.

---

## **Project Steps**

### 1. **Dataset Preparation**
   - Download the **Fashion MNIST Dataset** from the link provided above.
   - Images are stored in a **2D array of size n×784**.
   - Labels are stored as a **1D array** of class indices.

### 2. **Normalization**
   - Normalize the pixel values of the images by dividing them by **255**. This scales the values to the range [0, 1] to improve training efficiency.

### 3. **Cross-Entropy Loss Implementation**
   - Implement the cross-entropy loss function with weight regularization as shown above.
   - Regularize only the weights **w** and leave biases **b** unregularized.

### 4. **Stochastic Gradient Descent (SGD)**
   - Implement the **SGD algorithm** to minimize the loss function.
   - Split the dataset into **training** and **validation** sets.
   - Tune the learning rate, regularization constant (\(\alpha\)), and batch size as required.

### 5. **Performance Evaluation**
   - After tuning the hyperparameters, evaluate the model on the **test set**.
   - Report:
     - **Unregularized Cross-Entropy Loss**
     - **Percentage of Correctly Classified Examples**


