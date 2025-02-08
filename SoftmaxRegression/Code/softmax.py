import numpy as np

training_data = np.reshape(np.load("fashion_mnist_train_images.npy"), (-1, 28*28))
training_labels = np.load("fashion_mnist_train_labels.npy")

num_datapoints = training_data.shape[0]

split_index = int(0.8*num_datapoints)

indices = np.arange(num_datapoints)
np.random.shuffle(indices)

train_indices = indices[:split_index]
val_indices = indices[split_index:]

X_train, X_val = training_data[train_indices], training_data[val_indices]   
y_train, y_val = training_labels[train_indices], training_labels[val_indices]

X_train = X_train / 255.0
X_val = X_val / 255.0

input_size = 28*28
output_size = 10 # num_classes

def softmax(Z):
    # to avoid large exponent values, we will use the deviation idealogy of subtracting each value with the max value , thereby keeping the order preserved and since it will be normalised , result wont matter
    
    exponent_z = np.exp(Z - np.max(Z,axis=1,keepdims=True))
    
    prediction = exponent_z/(np.sum(exponent_z,axis=1,keepdims=True))
    
    return prediction

def loss(y_label,y_pred,W,alpha):
    
    batch_s = y_label .shape[0]  # to divide for average loss over batch
    
    prob = -np.log(y_pred[range(batch_s),y_label])  # for each sample we source the true label from y_label and compute the log of corresponding label value from y_pred
    
    loss = np.sum(prob) / batch_s
    
    reg_loss = alpha/2 * np.sum(np.square(W))
    
    batch_loss = loss + reg_loss
    
    return batch_loss

def gradient(X_batch,Y_batch,W,alpha,pred,B,learning_rate):
    
    batch_s = X_batch.shape[0]  # here its 64
    
    predi = pred
    
    predi[range(batch_s),Y_batch] -=1 # subtract each predicted true class label probability from 1 to compute the loss
    
    predi /= batch_s    # we compute the average loss per sample
    
    weight_grad = np.dot(X_batch.T,predi) + alpha * W
    bias_grad = np.sum(predi, axis = 0, keepdims=True)
    
                
    # update weights and bias using gradient descent
    W -= learning_rate * weight_grad 
    B -= learning_rate * bias_grad
    
    return W, B

def train_softmax(X_train,y_train,num_epochs,batch_size,learning_rate,alpha):
    
    num_tr_samples = X_train.shape[0] 
    tr_indices = np.arange(num_tr_samples)
    
    # initialize weight
    W = np.random.randn(input_size,output_size) * 0.01  # to avoid large initializations as its random  # for each image pixel spanning across 10 classes
    B = np.zeros((1,output_size))
    
    for epoch in range(num_epochs):
        np.random.shuffle(tr_indices)

        for num in range(0,num_tr_samples,batch_size):

                batch_index = tr_indices[num: num + batch_size]


                x_batch = X_train[batch_index]
                y_batch = y_train[batch_index]

                Z = np.dot(x_batch,W) + B   # batcsizex10

                pred = softmax(Z)

                batch_loss = loss(y_batch,pred,W,alpha)

                # next we will update the weights and bias

                W,B = gradient(x_batch,y_batch,W,alpha,pred,B,learning_rate)



        # print(f'Epoch {epoch+1}/{num_epochs}, Loss: {batch_loss}')
        
    return W,B


# Lets tune our hyperparameters

def validation(X_train,y_train,X_val,y_val):
    
    learning_rates = [1e-4,1e-3,1e-2] 
    mini_batch_sizes = [32, 64,128]
    num_epochs_testing = [50, 100,150]
    alpha = [1e-2,1e-1]
    
    best_accuracy = 0  # setting mse to positive infinity to ensure the first mse calculated becomes the default best value after first iteration and gets updated in the process
    best_hyperparams = {}   # dictionary to store the three HP parameters
    best_weights, best_bias = None, None
    
    for rate in learning_rates:
        for a in alpha:
            for batch in mini_batch_sizes:
                for epoch in num_epochs_testing:
                    
                    weights, bias = train_softmax(X_train,y_train,epoch,batch,rate,a)
                    
                    
                    
                    Z = np.dot(X_val, weights) + bias
                    A = softmax(Z)
                    y_pred = np.argmax(A, axis=1)
                    
                    accuracy = np.mean(y_pred == y_val)
                        
                    # print(f"Num_Epoch {epoch}, Batch_size {batch}, Learning_rate {rate}, Alpha {a}, f'Test accuracy: {accuracy * 100:.2f}%'")
                    
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_hyperparameters = {'num_epochs': epoch,'learning_rate': rate,'mini_batch': batch, 'Alpha': {a}}
                        best_weights,best_bias = weights,bias
                        
    return best_hyperparameters,best_weights,best_bias,best_accuracy


best_hyp,best_weights,best_bias,best_acc= validation(X_train,y_train,X_val,y_val)

np.save("best_model_weights3.npy", best_weights)
np.save("best_model_bias3.npy", best_bias)
np.save("best_model_hyperparameter3",best_hyp)
np.save("best_model_accuracy",best_acc)

def test(X,W,B):
    Z = np.dot(X, W) + B
    A = softmax(Z)
    return np.argmax(A, axis=1)

if __name__ == '__main__':
    test_weights = np.load("best_model_weights3.npy")
    test_bias = np.load("best_model_bias3.npy")

    testing_data = np.reshape(np.load("fashion_mnist_test_images.npy"), (-1, 28*28))
    testing_labels= np.load("fashion_mnist_test_labels.npy")
    
    X_test = testing_data / 255.0
    y_test = testing_labels
    y_pred = test(X_test,test_weights,test_bias)
    accuracy = np.mean(y_pred == y_test)
    print(f'Test accuracyafter hyper-parameter: {accuracy * 100:.2f}%')
