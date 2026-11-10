import numpy as np
from numpy.typing import NDArray
from pathlib import Path
from urllib.request import urlretrieve

#We are going to learn to do digit classification from MNIST dataset. 

#Structure:
#Input (784) -> Linear (512) -> ReLU -> Linear (128) -> ReLU -> Linear (10) -> Sigmoid -> Output (1)

#with batching and dropout + Adam

class MLP:
    
    def __init__(self):
        self.w1=np.random.randn(784, 512) * np.sqrt(2 / 784)        #He initialization
        self.b1=np.zeros((1,512))
        
        self.w2=np.random.randn(512, 128) * np.sqrt(2 / 512)
        self.b2=np.zeros((1,128))
        
        self.w3=np.random.randn(128, 10) * np.sqrt(2 / 128)
        self.b3=np.zeros((1,10))
        
    
    def relu(self, val: NDArray):
        val1=val.copy()
        for i in range(val1.shape[0]):
            for j in range(val1.shape[1]):
                val1[i,j]=max(0,val1[i,j])
        return val1
    
    
    def forward(self,x: NDArray):
        self.a1=x @ self.w1 + self.b1
        self.N1=self.relu(self.a1)
        
        self.a2=self.N1 @ self.w2 + self.b2
        self.N2=self.relu(self.a2)
        
        self.logits=self.N2 @ self.w3 + self.b3
        
        #softmax
        shifted=self.logits-np.max(self.logits, axis=1, keepdims=True)
        exp_logits=np.exp(shifted)
        self.y = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        
        return self.y
    
    def backward(self, x: NDArray, target: NDArray):
        self.der_y=(self.y-target)/target.shape[0]
        
        self.der_x3=(self.der_y @ self.w3.T)
        self.der_w3=(self.N2.T @ self.der_y)
        self.der_b3 = np.sum(self.der_y, axis=0, keepdims=True)
        
        
        der_z2=self.der_x3 * (self.N2 > 0)      #relu
        self.der_x2=(der_z2 @ self.w2.T)
        self.der_w2=(self.N1.T @ der_z2)
        self.der_b2=np.sum(der_z2, axis=0, keepdims=True)
        
        der_z1=self.der_x2 * (self.N1 > 0)      #relu
        self.der_x1=(der_z1 @ self.w1.T)
        self.der_w1=(x.T @ der_z1)
        self.der_b1=np.sum(der_z1, axis=0, keepdims=True)
        
    def step(self,learning_rate):
        self.w1 -= self.der_w1 * learning_rate
        self.w2 -= self.der_w2 * learning_rate
        self.w3 -= self.der_w3 * learning_rate

        self.b1 -= self.der_b1 * learning_rate
        self.b2 -= self.der_b2 * learning_rate
        self.b3 -= self.der_b3 * learning_rate


def load_data():
    with np.load("data/mnist.npz") as data:
        X_train = data["x_train"]
        y_train_old = data["y_train"]
        X_test = data["x_test"]
        y_test_old = data["y_test"]
    
    X_train=X_train.reshape(-1,28*28).astype(np.float64)/255.0
    X_test=X_test.reshape(-1,28*28).astype(np.float64)/255.0
    
    y_train=np.zeros((y_train_old.shape[0],10))
    y_test=np.zeros((y_test_old.shape[0],10))
    
    for i in range(y_train_old.shape[0]):
        y_train[i,y_train_old[i]]=1
    
    
    for i in range(y_test_old.shape[0]):
        y_test[i,y_test_old[i]]=1
    
    
    return X_train,X_test,y_train,y_test

X_train, X_test, Y_train, Y_test=load_data()


#training loop
model=MLP()

def accuracy(pred,target):
    correct=0
    total=0
    for i,j in zip(pred,target):
        res=np.argmax(i)
        true_res=np.argmax(j)
        if(res==true_res):
            correct+=1
        total+=1
    print(f"Accuracy: {correct/total}")
BATCH_SIZE=60
EPOCHS=10
for i in range(EPOCHS):
    for batch in range(int(X_train.shape[0]/BATCH_SIZE-1)):
        features=X_train[batch*BATCH_SIZE:(batch+1)*BATCH_SIZE,:]
        labels=Y_train[batch*BATCH_SIZE:(batch+1)*BATCH_SIZE,:]
        pred=model.forward(features)
        model.backward(features,labels)
        model.step(learning_rate=0.01)
    
    

    print(f"Epoch {i}")
    pred=model.forward(X_train)
    labels=Y_train
    accuracy(pred,labels)


#test
pred=model.forward(X_test)
correct=0
total=0
for i,j in zip(pred,Y_test):
    res=np.argmax(i)
    true_res=np.argmax(j)
    if(res==true_res):
        correct+=1
    total+=1


print(f"Accuracy: {correct/total}")
