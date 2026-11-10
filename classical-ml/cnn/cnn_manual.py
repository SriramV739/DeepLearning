import numpy as np
from numpy.typing import NDArray
from pathlib import Path
from urllib.request import urlretrieve



#We are going to make a CNN to classify CIFAR RGB images into categories. 
#input: Batchx3x32x32
#Layer 1: 32 channels, Kernel 3x3, stride 1, padding 1 -> 32x32x32
#Layer 2: Batch norm + Relu
#Layer 3: 64 channels, Kernel 3x3, stride 1, padding 1 -> 64x32x32
#Layer 4: BN+Relu+ MaxPool -> 64x16x16
#Layer 5: 128 channels out
#Max pool -> 128x8x8
#Avg pool-> 128x2x2
#Dropout
#Linear -> 10





class CNN:
    def __init__(self):
        self.layer1_conv=np.random.randn(32,3,3,3) * np.sqrt(2 / 27) 
        self.gamma1=np.ones(32)
        self.beta1=np.zeros(32)
        
        self.layer2_conv=np.random.randn(64,32,3,3) * np.sqrt(2 / 32*3*3) 
        self.gamma2=np.ones(32)
        self.beta2=np.zeros(32)
        
        self.layer3_conv=np.random.randn(128,64,3,3) * np.sqrt(2 / 64*3*3) 
        self.gamma3=np.ones(64)
        self.beta3=np.zeros(64)
        
        
        self.layer4_linear=np.random.randn(128*2*2, 10)
        
    def relu(self, val: NDArray):
        val1=val.copy()
        for i in range(val1.shape[0]):
            for j in range(val1.shape[1]):
                for k in range(val.shape[2]):
                    for l in range(val.shape[3]):
                        val1[i,j,k,l]=max(0,val[i,j,k,l])

        return val1


    def pad(self,val: NDArray):
        ret=np.zeros(val.shape[0],val.shape[1]+2,val.shape[2]+2, val.shape[3])
        for ind in range(ret.shape[0]):
            for i in range(val.shape[1]):
                for j in range(val.shape[2]):
                    for k in range(val.shape[3]):
                        ret[ind,i,j+1,k+1]=val[ind,i,j,k]
            
        return ret

    def convolve(self,val: NDArray, kernel: NDArray):
        ret=np.zeros(val.shape[0],kernel.shape[0], val.shape[2]-2, val.shape[3]-2)
        for ind in range(val.shape[0]):
            for i in range(1,val.shape[1]-1):
                for j in range(1,val.shape[2]-1):
                    for k in range(kernel.shape[0]):
                        ret[ind,k]=kernel[k]*(val[ind,:,i-1:i+1,j-1:j+1])
        return ret
    
    def max_pool(self,val: NDArray):
        ret=np.zeros(val.shape[0], val.shape[1],val.shape[2]//2, val.shape[3]//2)
        for i in range(ret.shape[0]):
            for j in range(ret.shape[1]):
                for k in range(ret.shape[2]):
                    for l in range(ret.shape[3]):
                        ret[i,j,k,l]=np.max(val[i,j,2*k : 2*k+1 , 2*l : 2*l+1])
        return ret

    def batch_norm(self,val, alpha, gamma):
        ret=val.copy()
        for channel in range(val.shape[1]):
            mean=0
            total=0
            for i in range(val.shape[0]):
                for j in range(val.shape[2]):
                    for k in range(val.shape[3]):
                        mean+=val[i,channel,j,k]
                        total+=1
            
            mean/=total
            
            var=0
            for i in range(val.shape[0]):
                for j in range(val.shape[2]):
                    for k in range(val.shape[3]):
                        sd+=(val[i,channel,j,k]-mean)**2

            var/=total
            for i in range(val.shape[0]):
                for j in range(val.shape[2]):
                    for k in range(val.shape[3]):
                        ret[i,channel,j,k]=(val[i,channel,j,k]-mean)/(np.sqrt(var) + 1e-5)


    def forward(self, x:NDArray):
        padded_1=self.pad(x)
        

        
    