import nibabel as nib
import os
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import csv

base_dir = '/home/fhd/projects/.DATASETS/HealthyControls/'
ind_dir = os.path.join(base_dir,'C01')
rec_path = os.path.join(ind_dir,'left_foot_trial_21.nii')
INPUT = 'input.csv'

#defing necessary functoins
def read_single(img_path:str) -> np.memmap:
    '''function that gets the path of a single record and returns the memmap array in shape of (36,20,:)'''
    img = nib.load(img_path)
    try:
        data = img.get_fdata()
        if not (data.shape[1] <20 or data.shape[0] < 35):
            return img.get_fdata()[:35,:20,:]
    except :
        return None

def read_all(all_path:list) -> pd.DataFrame:
    '''get list of paths and returns of Y memmap of all records shape of (36,20,:)'''
    Y = np.zeros((35,20,0)) # initiating Y 
    #reading all images
    for path in all_path:
        arr = read_single(os.path.join(ind_dir,path))
        if type(arr)==np.memmap: Y = np.append(Y,arr,axis=2) #appending to Y
    return Y
    
def preprocess(memmap:np.memmap) -> pd.DataFrame:
    memmap = memmap.reshape(-1, memmap.shape[-1])
    return pd.DataFrame(memmap).T

def create_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=(10,)))
    model.add(tf.keras.layers.Dense(16,activation='relu'))
    model.add(tf.keras.layers.Dense(32,activation='relu'))
    model.add(tf.keras.layers.Dense(64,activation='relu'))
    model.add(tf.keras.layers.Dense(128,activation='relu'))
    model.add(tf.keras.layers.Dense(256,activation='relu'))
    model.add(tf.keras.layers.Dense(512,activation='relu'))
    model.add(tf.keras.layers.Dense(700,activation='linear'))
    model.compile(optimizer='adam', loss='mse',metrics=['MeanSquaredError',])
    return model

def get_all_path( Left:bool =True, ex=None,inc=None) -> list:
    '''if you want the left foot then put True else Flase, parameter ex excludes from output'''
    all_path=[]
    for folder in list(os.walk(base_dir)):
        for filename in folder[2]:
            if Left:
                if 'left' in filename:
                    all_path.append(os.path.join(folder[0],filename))
            else:
                if 'right' in filename:
                    all_path.append(os.path.join(folder[0],filename))
    if ex: all_path= list(filter(lambda x: ex not in x,all_path))
    elif inc: all_path= list(filter(lambda x: inc in x,all_path))
    return all_path

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

def convert_2dIndex_to_1d(index):
    shape = (35,20)
    x,y = index
    _,Y = shape
    return x*Y+y
        

#making Y dataset
all_path = get_all_path(ex='C31')
Y = read_all(all_path)
Y = preprocess(Y)

#making X dataset
selected_nodes_2d =[(14,4),(11,4),(5,21),(3,22),(10,27),(7,32),(6,30),(9,32),(12,27),(6,23)]
selected_nodes_1d = list(map(convert_2dIndex_to_1d,selected_nodes_2d))
X = Y[selected_nodes_1d]

# Splitting datasets
X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size=0.2, shuffle =False)

model = create_model()

model.fit(
    x=X_train, y=y_train, batch_size=64, epochs=5,
    validation_split=0.2, validation_data=None, shuffle=True,
    workers=2, use_multiprocessing=True
)

#making plot dataset
test_path = get_all_path(inc='C31')
Yplot = read_all(test_path)
Yplot = preprocess(Yplot)
Xplot = Yplot[selected_nodes_1d]

#predicting results for Xplot
results = model.predict(Xplot)

#predicting result for ploting
result = results.T
result = result.reshape(35,20,result.shape[-1])

#making animation
x,y,_ = read_input(INPUT)
animate(x,y,result,'new_animation_10_nodes_5_epoch_700.mp4')

#saving the model
model.save('model.hdf5')