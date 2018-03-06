#Author=====Anshuk Uppal
'''Max accuracy achieved by this model is nearly 50% right now'''


import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras.datasets import cifar10
from keras.utils import np_utils
from keras.optimizers import SGD
from keras.optimizers import Adadelta
from keras.optimizers import Adam
from keras from keras import regularizers

#load data
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

#normalizing the inputs as the image intensities range from 0-255
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train / 255.0
X_test = X_test / 255.0

#transform into binary matrix // one hot encoding
y_train=np_utils.to_categorical(y_train)
y_test=np_utils.to_categorical(y_test)
num_classes=y_test.shape[1]

#reshaping the the data into a matrix form and each row having one image
X_train=np.reshape(X_train,(np.shape(X_train)[0],32*32*3))
X_test=np.reshape(X_test,(np.shape(X_test)[0],32*32*3))

#making the model
model=Sequential()
model.add(Dense(100,input_dim=3072,activation='relu',kernel_regularizer=regularizers.l2(0.0001)))##first layer needs to have  the input dimensions and use of rectifier activation function
model.add(Dense(70,activation='relu'))##,kernel_regularizer=regularizers.l2(0.01)))
model.add(Dense(50,activation='relu'))##,kernel_regularizer=regularizers.l2(0.01)))
model.add(Dense(30,activation='relu'))
model.add(Dense(num_classes,activation='softmax'))##last layer
##Various optimizers for the neural network
#model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#sgd = SGD(lr=0.02, decay=0.01/25, momentum=0.9, nesterov=False)
#model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])
#Ada=Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
#model.compile(loss='categorical_crossentropy',optimizer=Ada,metrics=['accuracy'])
adm=Adam(lr=0.00115, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0, amsgrad=False)#increased the learning rate from default
model.compile(loss='categorical_crossentropy', optimizer=adm, metrics=['accuracy'])
print(model.summary())

##Training the model
model.fit(X_train,y_train,epochs=25,batch_size=32)

##evaluation
eva=model.evaluate(X_test,y_test,verbose=0)
print("Accuracy is %.2f"%(eva[1]*100))

