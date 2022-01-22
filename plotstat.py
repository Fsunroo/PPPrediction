import os
import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd

i=40

base_dir = '/home/fhd/projects/frelnc/signal/cnnmodel/data/'
rec_path = os.path.join(base_dir,'Patient_03_zool2_links.asc')
INPUT_path = '/home/fhd/projects/frelnc/signal/compress1/input.csv'

def read_input(INPUT):
    x_input,y_input,p=[],[],[]
    with open(INPUT, newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            x_input.append(row[0])
            y_input.append(row[1])
            p.append(row[2])
    x_input = np.array(x_input[1:],dtype=float)
    y_input = np.array(y_input[1:],dtype=float)
    p = np.array(p[1:],dtype=float)
    return x_input,y_input,p
 
def read_single(rec_path:str,Left=True) -> pd.DataFrame:
    f = open(rec_path,'r').readlines()
    f=[i.strip() for i in f][10:]
    f=[i.split('\t') for i in f]
    df = pd.DataFrame(f).drop(0,axis=1)

    df=df.astype(float)
    if Left:
        df= df[[i for i in range(1,100)]]
    else:
        df= df[[i for i in range(100,199)]]

    for i in range(len(df)):
        if df.loc[i].any()==0:
            df = df.drop(i,axis=0)
    df = df.reset_index().drop('index',axis=1)
    return df

x,y,_ = read_input(INPUT_path)
result = read_single(rec_path)

fig = plt.figure(figsize=(3,6))
plt.tricontourf(x,y,result.loc[i],cmap='Reds')
plt.plot(x,y,'k.')
plt.title(i)
plt.show()