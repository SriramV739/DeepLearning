import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F



class LogisticRegression(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear=nn.Linear(input_dim,1)
        
        
    
    def forward(self,x):
        return torch.sigmoid(self.linear(x))




model=LogisticRegression(6)



df=pd.read_csv("logistic_regression_data.csv")
num_rows=df.shape[0]

train_df=df.iloc[:int(num_rows*0.8)]
test_df=df.iloc[int(num_rows*0.8):]


label="label"
# train_cols=["feature_1","feature_2","feature_3","feature_4","feature_5","feature_6"]
train_cols=[x for x in df.columns if x!=label]

train_x=train_df[train_cols].values
train_y=train_df[label].values


test_x=test_df[train_cols].values
test_y=test_df[label].values

# need to add .values() to above??


train_x_tensor=torch.tensor(train_x,dtype=torch.float32)
train_y_tensor=torch.tensor(train_y,dtype=torch.float32).view(-1,1)


test_x_tensor=torch.tensor(test_x,dtype=torch.float32)
test_y_tensor=torch.tensor(test_y,dtype=torch.float32).view(-1,1)


loss_fn=nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
for epoch in range(10000):
    logits=model(train_x_tensor)
    loss=loss_fn(logits,train_y_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print(epoch, loss.item())



pred=model(test_x_tensor)
loss=loss_fn(pred,test_y_tensor)

print(loss.item())
    
    


model.eval()

with torch.no_grad():
    probabilities = model(test_x_tensor)
    predictions = (probabilities >= 0.5).float()

    test_loss = loss_fn(probabilities, test_y_tensor)
    accuracy = (predictions == test_y_tensor).float().mean()

print("Test loss:", test_loss.item())
print("Test accuracy:", accuracy.item())




# x_train=



