import pandas as pd
import numpy as np


df=pd.read_csv("data.csv")

num_rows=df.shape[0]


X = df.iloc[:int(num_rows*.8), 1:-1]


Y= df.iloc[:int(num_rows*.8),-1]

x_arr=X.to_numpy()

x_t=x_arr.T

#normal equation
train_weights=np.linalg.inv((x_t@x_arr))@x_t@Y


print(train_weights)




X_test=df.iloc[int(num_rows*.8):, 1:-1]
print(X_test.iloc[0].shape[0])
Y_test= df.iloc[int(num_rows*.8):,-1]



error=0
count=0


for index,row in X_test.iterrows():
    row_arr=row.to_numpy()
    val=row_arr@(train_weights.T)
    error+=((val-Y_test.iloc[count])/Y_test.iloc[count])**2
    count+=1
    

print(error/count)



    
    






