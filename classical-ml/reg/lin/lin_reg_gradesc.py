import pandas as pd
import numpy as np
num_epochs=2
alpha=0.05


def normalize_dataframe(df):
    mean = df.mean()
    std = df.std().replace(0, 1)

    normalized_df = (df - mean) / std

    return normalized_df, mean, std


df=pd.read_csv("/Volumes/SeagateSSD/ml-work/data.csv")
print(df.shape[0])
num_rows=int(df.shape[0]*0.8)
num_columns=10 #not including label



#normalize if it doesnt work




x_train=df.iloc[:int(num_rows),1:-1]
y_train=df.iloc[:int(num_rows),-1]



x_train, mean, std = normalize_dataframe(x_train)

x_train[num_columns]=1 #need to add extra column of just 1's at the end

x_test=df.iloc[int(num_rows):,1:-1]
y_test=df.iloc[int(num_rows):,-1]



x_test = (x_test - mean) / std


x_test[num_columns]=1 #need to add extra column of just 1's at the end 





feat_guess=np.zeros(num_columns+1)


print(feat_guess)
for i in range(num_epochs):
    for feature in range(num_columns+1):
        sum2=0
        for row in range(len(x_train)):
            sum1=(x_train.iloc[row].to_numpy()) @ feat_guess - y_train.iloc[row]
            sum1*=x_train.iloc[row,feature]
            sum2+=sum1
    
        sum2*=alpha
        feat_guess[feature]-=sum2/len(x_train)



# error analysis


error=0
count=0


for index,row in x_test.iterrows():
    row_arr=row.to_numpy()
    val=row_arr@(feat_guess.T)
    error+=((val-y_test.iloc[count])/y_test.iloc[count])**2
    count+=1
    

print(error/count)
print(feat_guess)    



