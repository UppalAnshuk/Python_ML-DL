import numpy
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten
from keras.layers.convolutional import Conv3D,MaxPooling3D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')

(X_train, Y_train), (x_test, y_test) = cifar10.load_data()
X_train=X_train.astype('float32')
x_test=x_test.astype('float32')
X_train=X_train/255.0
x_test=x_test/255.0

Y_train = np_utils.to_categorical(Y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]
















###########################################################
''' model1=Sequential()
>>> model1.add(Conv3D(20,(3,3,3),strides=(1,1,1),input_shape=(50000,32,32,3),padding="valid",dilation_rate=0))
>>> model1.add(Dropout(0.2))
>>> model1.add(Conv3D(40,(3,3,3),strides=(1,1,1),padding="valid",dilation_rate=0,activation='relu',use_bias=True)
... 
KeyboardInterrupt
>>> model1.add(Conv3D(40,(3,3,3),strides=(1,1,1),padding="valid",dilation_rate=0,activation='relu',use_bias=True))
>>> model1.add(Dropout(0.4))
>>> model1.add(Conv3D(20,(5,5,3),strides=(1,1,1),padding="valid",dilation_rate=0,activation='relu',use_bias=True))
>>> mode2=Sequential()
>>> mode2.add(Conv3D(20,(3,3,3),strides=(1,1,1),input_shape=(50000,32,32,3),padding="valid",dilation_rate=0))
>>> mode2.add(Dropout(0.2))'''
model3=Sequential()
model3.add(Conv3D(20,(3,3,3),strides=(1,1,1),input_shape=(500,3,32,32),padding="same",data_format="channels_first",use_bias=False))
model3.add(Dropout(0.2))
model3.add(Conv3D(40,(1,7,7),strides=(1,1,1),padding="valid",activation='relu',use_bias=True))
model3.add(Dropout(0.4))
model3.add(Conv3D(40,(1,7,7),strides=(1,1,1),padding="valid",activation='relu',use_bias=True))
model3.add(MaxPooling3D(pool_size=(1,2,2),padding='valid'))
model3.add(Flatten())
model3.add(Dense(128, activation='relu'))
model3.add(Dropout(0.5))
model3.add(Dense(num_classes, activation='softmax'))
# Compile model
epochs = 25
lrate = 0.01
decay = lrate/epochs
sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
model3.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
print(model3.summary())
########################
