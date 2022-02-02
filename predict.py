import tensorflow as tf
from utils import *

test_path = '/home/fhd/projects/frelnc/signal/cnnmodel/Patient_03_zool2_links.asc'
model_path ='/home/fhd/projects/frelnc/signal/cnnmodel/model99.hdf5'
input_path = '/home/fhd/projects/frelnc/signal/compress1/input.csv'



x,y,selected_nodes_1d = get_selected_nodes_1d(input_path)
test = read_single(test_path)
test_x = test[selected_nodes_1d]


model = tf.keras.models.load_model(model_path)
result = model.predict(test_x)
acc = mse(test.to_numpy(),result)
print('mse is: ',acc)

animate(x,y,result,'99predicted-bymodelV0.0.gif')
print('your prediction result is at output folder')
