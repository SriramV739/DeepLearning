import random
import pandas as pd
import numpy as np
num_columns=10
num_rows=10

columns=[]
for i in range(num_columns+1):
    columns.append(str(i))


df=pd.DataFrame(columns=columns)

for i in range(num_rows):
    
    row=[]
    
    for j in range(num_columns+1):
        row.append(random.randint(0,100000))
    
    
    df.loc[len(df)]=row
    


weights=np.random.uniform(0, 10, size=num_columns)



last_col=df.columns[-1]
df[last_col]=df[last_col].astype(float)

for index,row in df.iterrows():
    percent_off=random.uniform(0,5)
    # percent_off=0
    feat=row.iloc[:-1]
    val=feat@weights
    
    up=random.random()
    if(up>=0.5):
        val*=(100+percent_off)
    else:
        val*=(100-percent_off)
    
    val/=100

    df.at[index,last_col]=val



df.to_csv("/Users/sriram/projects/ML/impl/lin_reg/data_smaller.csv")
np.savetxt("weights.csv",weights,delimiter=",")
    


