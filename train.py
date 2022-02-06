from sklearn.model_selection import train_test_split
from utils import *

base_dir = '/home/fhd/projects/frelnc/signal/cnnmodel/data/'
input_path = '/home/fhd/projects/frelnc/signal/compress1/input.csv'

x,y,selected_nodes_1d = get_selected_nodes_1d(input_path)
Y = read_all(base_dir)
X = Y[selected_nodes_1d]
X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size=0.2, shuffle =False)

model = create_model()

model.fit(
    x=X_train, y=y_train, batch_size=64, epochs=30,
    validation_split=0.2, validation_data=None, shuffle=True,
    workers=2, use_multiprocessing=True
)
print('model trained successfully') 
if not os.path.exists('output'):
    os.makedirs('output')
model.save(os.path.join('output','model99.hdf5'))
print('Model saved successfully')
print('\n\n\n\nEvaluating model...')
res = model.evaluate(X_test,y_test)

print_result(res)

print('\n\n\n\n\nAnimating the grandTruth...')
animate(x,y,Y,'99sens.gif',max=350)
print('.gif file of your data is created at output folder \nNote that it is grand truth for predicting your own please run predict.py')