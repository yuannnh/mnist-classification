import numpy as np
import matplotlib.pyplot as plt
import time
from mnist import MNIST
from FullyConnectedNet import *
from solver import Solver


data_dir = '../datasets/python-mnist/data'
train_num = 60000
test_num = 10000
img_size = 28
print('____Running fully connected network____')
def loadTrainData():
    mndata = MNIST(data_dir)
    images, labels = mndata.load_training()
    X = np.array(images).reshape(train_num,img_size,img_size)
    Y = np.array(labels)   
    return X, Y

def loadTestData():
    mndata = MNIST(data_dir)
    images, labels = mndata.load_testing()
    X = np.array(images).reshape(test_num,img_size,img_size)
    Y = np.array(labels)  
    return X, Y

def load_data():
    print("loading data")
    X_train, Y_train = loadTrainData()
    X_test, Y_test = loadTestData()
    data = {}
    data['X_train'] = X_train
    data['y_train'] = Y_train
    data['X_val'] = X_test
    data['y_val'] = Y_test
    print(type(data))
    return data

data = load_data()

num_train = 60000
mydata = {
  'X_train': data['X_train'][:num_train],
  'y_train': data['y_train'][:num_train],
  'X_val': data['X_val'],
  'y_val': data['y_val'],
}

#solvers = {}

print('running with ', 'sgd_momentum')
print('learning rate: ',1e-2)
model = FullyConnectedNet([100, 100, 100, 100, 100], weight_scale=5e-2)

solver = Solver(model, mydata,
              num_epochs=15, batch_size=100,
              update_rule='sgd_momentum',
              optim_config={
                'learning_rate': 1e-2,
              },
              verbose=True)
#solvers[update_rule] = solver
solver.train()


# ______plot______

plt.subplot(3, 1, 1)
plt.title('Training accuracy')
plt.xlabel('Epoch')
plt.plot(solver.train_acc_history)



plt.subplot(3, 1, 2)
plt.title('Validation accuracy')
plt.xlabel('Epoch')
plt.plot(solver.val_acc_history)

plt.show()



