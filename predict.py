import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
from utils import *

test_path = '/home/fhd/projects/frelnc/signal/cnnmodel/data/Patient_03_zool2_links.asc'
model_path ='/home/fhd/projects/frelnc/signal/cnnmodel/output/model99.hdf5'
input_path = '/home/fhd/projects/frelnc/signal/compress1/input.csv'



x,y,selected_nodes_1d = get_selected_nodes_1d(input_path)
test = read_single(test_path)
test_x = test[selected_nodes_1d]


model = tf.keras.models.load_model(model_path)
print('model loaded successfully')
print('prediction started...')
result = model.predict(test_x)
print('prediction Done')
ms = mse(test.to_numpy(),result)
ma = mae(test.to_numpy(),result)
rm = rmse(test.to_numpy(),result)
print_result([ms,ms,ma,rm])

print('\n\n\nAnimating the predicted data...')
animate(x,y,result,'99predicted-bymodelV0.0.gif')
print('Your prediction result is in output folder')
