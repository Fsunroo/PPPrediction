import tensorflow as tf
from main import read_single,selected_nodes_1d,animate,read_input

test_path = '/home/fhd/projects/frelnc/signal/cnnmodel/Patient_03_zool2_links.asc'
model_path ='/home/fhd/projects/frelnc/signal/cnnmodel/model99.hdf5'
x,y,_ = read_input('/home/fhd/projects/frelnc/signal/compress1/input.csv')

test = read_single(test_path)
test_x = test[selected_nodes_1d]


model = tf.keras.models.load_model(model_path)
result = model.predict(test_x)

animate(x,y,result,'99predicted-bymodelV0.0.gif')

