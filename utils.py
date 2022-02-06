import os
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import pandas as pd
import csv
import matplotlib.pyplot as plt
import matplotlib.animation as animation

#defining necessary functions
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

def read_all(path:str) -> pd.DataFrame:
    df= pd.DataFrame()
    for file in os.listdir(path):
        if file.endswith('asc'):
            df = df.append(read_single(os.path.join(path,file)))
    return df.reset_index().drop('index',axis=1)

def create_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=(12,)))
    model.add(tf.keras.layers.Dense(16,activation='relu'))
    model.add(tf.keras.layers.Dense(32,activation='relu'))
    model.add(tf.keras.layers.Dense(64,activation='relu'))
    model.add(tf.keras.layers.Dense(99,activation='linear'))
    model.compile(optimizer='adam', loss='mse',metrics=['MeanSquaredError','MeanAbsoluteError','RootMeanSquaredError'])
    return model

def parse_result(result:list) -> dict:
    result_dict = {}
    result_dict['loss'] = result[0]
    result_dict['MeanSquaredError'] = result[1]
    result_dict['MeanAbsoluteError'] = result[2]
    result_dict['RootMeanSquaredError'] = result[3]
    return result_dict

def mse(y_true,y_pred):
    difference_array = np.subtract(y_true, y_pred)
    squared_array = np.square(difference_array)
    mse = squared_array.mean()
    return mse

def mae(y_true,y_pred):
    difference_array = np.subtract(y_true, y_pred)
    abs_array = np.absolute(difference_array)
    mae = abs_array.mean()
    return mae

def rmse(y_true,y_pred):
    return np.sqrt(mse(y_true,y_pred))

def convert_2dIndex_to_1d(index):
    shape = (35,20)
    x,y = index
    _,Y = shape
    return x*Y+y

def read_input(INPUT):
    x_input,y_input,p=[],[],[]
    with open(INPUT, newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            x_input.append(row[0])
            y_input.append(row[1])
            p.append(row[2])
        f.close()
    x_input = np.array(x_input[1:],dtype=float)
    y_input = np.array(y_input[1:],dtype=float)
    p = np.array(p[1:],dtype=float)
    return x_input,y_input,p

def animate(x:np.array,y:np.array,result:pd.DataFrame,name:str,max=300):
    if not 'output' in os.listdir(): os.mkdir('output')
    if type(result)==pd.DataFrame:
        result = result.to_numpy()
    result=result[:max]
    fig, ax=plt.subplots(figsize=(3,6))
    p = [ax.tricontourf(x,y,result[0],cmap='Reds')]
    def update(i):
        for tp in p[0].collections:
            tp.remove()
        p[0] = ax.tricontourf(x,y,result[i],cmap='Reds') 
        return p[0].collections

    ani = animation.FuncAnimation(fig, update, blit=True, repeat=True)
    ani.save(os.path.join('output',name), fps=30)

def calculate_dist(pointa,pointb):
    xa,ya=pointa
    xb,yb=pointb
    return ((xa-xb)**2 + (ya-yb)**2)**0.5

def find_nearest(array, point,cordinates):
    idx = np.asarray([calculate_dist(cordinate,point) for cordinate in cordinates]).argmin()
    return array[idx]

def print_result(res):
    print('\n\nmoodel evalutaion result:')
    print ("{:<15} {:<20} {:<20} {:<20} {:<20}".format('Metrics','loss','MeanSqureError','MeanAbsoluteError','RootMeanSquaredError'))
    #for k, v in res.items():
    l, ms, ma,rm = res
    print ("{:<15} {:<20} {:<20} {:<20} {:<20}".format('Value', l, ms, ma, rm))

def get_selected_nodes_1d(input_path):
    x,y,_ = read_input(input_path)
    cordinates = [(i,j) for i,j in zip (x,y)]

    selected_nodes_2d =[(4.99,3.3),(3.66,3.3),(6.32,3.3),(4.99,4.6), #posterior
                        (7.6,12.4),(7.6,11.1),(6.2,11.1),           #Lateral
                        (4.68,12.4),                                #Median
                        (8.58,17.6),                                #AntroLateral
                        (1,18.9),                                   #AntroMedial
                        (1.55, 15.0),(3.1, 15.0),                   #Medial
                        ]                      
                        
    selected_nodes_1d = list(map(lambda x : cordinates.index(x),selected_nodes_2d))
    return x,y,selected_nodes_1d