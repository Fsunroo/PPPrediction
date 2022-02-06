from ast import Break
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
from utils import *


test_dir= os.path.join('predict')
model_path =os.path.join('output','model99.hdf5')
input_path = os.path.join('input.csv')

test_files = list(filter(lambda x : x.endswith('.asc'), [file for file in  os.listdir(test_dir)]))
if len (test_files )== 1: test_path = os.path.join(test_dir,test_files[0])
else:raise Exception('Please put only one file in predict folder')


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
