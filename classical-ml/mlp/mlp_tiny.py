import numpy as np

# We are going to do a 2 feature binary input space that is just trying to learn the XOR function. We will have the layers like this:
# Input(2)→Hidden 1(2)→Hidden 2(2)→Output(1)

# Hidden layer 1: ReLU
# Hidden layer 2: ReLU
# Output: sigmoid

class MLP:
    def __init__(self):
        #randomly initialize the weights
        self.w1=0.1
        self.w2=0.3
        self.w3=-0.4
        self.w4=0.1
        self.w5=-0.3
        self.w6=-0.2
        self.w7=-0.3
        self.w8=0.2
        self.w9=0.3
        self.w10=-0.2
        
        #biases
        self.b1=0.05
        self.b2=0.1
        self.b3=0.075
        self.b4=0.1
        self.b5=0.05
        
    def sigmoid(self,val):
        return 1/(1+np.exp(-1*val))
    
    # def relu(self,val):
    #     return max(0,val)
    
    
    #try leaky relu:
    def relu(self, val):
        return val if val > 0 else 0.01 * val
    
    def forward_pass(self, x1, x2):
        self.a1=(self.w1*x1 + self.w2*x2 + self.b1)
        self.N1=self.relu(self.a1)
        self.a2=(self.w3*x1 + self.w4*x2 + self.b2)
        self.N2=self.relu(self.a2)
        self.a3=(self.w5*self.N1 + self.w6*self.N2 + self.b3)
        self.N3=self.relu(self.a3)
        self.a4=(self.w7*self.N1 + self.w8*self.N2 + self.b4)
        self.N4=self.relu(self.a4)
        self.o=(self.w9*self.N3 + self.w10*self.N4 + self.b5)
        self.y=self.sigmoid(self.o)
        return self.y

    def MSEloss(pred,target):
        return ((pred-target)**2)/2
    
    def backward(self, x1, x2, target):
        # self.forward_pass(x1,x2)
        # loss=self.MSEloss(self.y,target)
        
        #calculating all the gradients by hand and hard encoding them
        
        der_y=self.y-target
        der_o=der_y*self.y*(1-self.y)                  # this is equaivalent to the actual chain rule calculation: der_y*(1/((1+np.exp(-(self.o)))**2))*np.exp()
        
        
        self.der_w9=der_o*self.N3
        self.der_w10=der_o*self.N4
        self.der_b5=der_o
        der_N3=der_o*self.w9
        der_N4=der_o*self.w10
        
        
        # der_a3=der_N3*(self.a3>0)
        # der_a4=der_N4*(self.a4>0)
        
        #new leaky relu stuff
        der_a3 = der_N3 * (1 if self.a3 > 0 else 0.01)
        der_a4 = der_N4 * (1 if self.a4 > 0 else 0.01)
        
        
        self.der_w5=der_a3*self.N1
        self.der_w6=der_a3*self.N2
        self.der_w7=der_a4*self.N1
        self.der_w8=der_a4*self.N2
        self.der_b3=der_a3
        self.der_b4=der_a4
        
        der_N1=der_a3*self.w5 + der_a4*self.w7
        der_N2 = der_a3*self.w6 + der_a4*self.w8
        
        
        # der_a1=der_N1*(self.a1>0)
        # der_a2=der_N2*(self.a2>0)
        der_a1 = der_N1 * (1 if self.a1 > 0 else 0.01)
        der_a2 = der_N2 * (1 if self.a2 > 0 else 0.01)
        
        self.der_w1=der_a1*x1
        self.der_w2=der_a1*x2
        self.der_w3=der_a2*x1
        self.der_w4=der_a2*x2
        self.der_b1=der_a1
        self.der_b2=der_a2
        
        
        
    def step(self, learning_rate):
        self.w1 -= self.der_w1 * learning_rate
        self.w2 -= self.der_w2 * learning_rate
        self.w3 -= self.der_w3 * learning_rate
        self.w4 -= self.der_w4 * learning_rate
        self.w5 -= self.der_w5 * learning_rate
        self.w6 -= self.der_w6 * learning_rate
        self.w7 -= self.der_w7 * learning_rate
        self.w8 -= self.der_w8 * learning_rate
        self.w9 -= self.der_w9 * learning_rate
        self.w10 -= self.der_w10 * learning_rate

        self.b1 -= self.der_b1 * learning_rate
        self.b2 -= self.der_b2 * learning_rate
        self.b3 -= self.der_b3 * learning_rate
        self.b4 -= self.der_b4 * learning_rate
        self.b5 -= self.der_b5 * learning_rate


m=MLP()
arr=[
    [0,0,0],
    [0,1,1],
    [1,0,1],
    [1,1,0]
]


import random
for epoch in range(100000):
    ind=random.randint(0,3)
    m.forward_pass(arr[ind][0],arr[ind][1])
    m.backward(arr[ind][0],arr[ind][1],arr[ind][2])
    m.step(0.01)


for i in arr:
    print(m.forward_pass(i[0],i[1]), i[2])
    # print(f"Bad: {i}")
        