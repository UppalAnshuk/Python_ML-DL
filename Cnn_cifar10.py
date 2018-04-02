import numpy
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten
from keras.layers.convolutional import Conv2D,MaxPooling2D
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

model3=Sequential()
model3.add(Conv2D(20,(3,3),strides=(1,1),input_shape=(3,32,32),padding="same",data_format="channels_first",use_bias=False))
model3.add(Dropout(0.2))
model3.add(Conv2D(40,(7,7),strides=(1,1),padding="valid",activation='relu',use_bias=True))
model3.add(Dropout(0.4))
model3.add(Conv2D(40,(7,7),strides=(1,1),padding="valid",activation='relu',use_bias=True))
model3.add(MaxPooling2D(pool_size=(2,2),padding='valid'))
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


# Fit the model
model3.fit(X_train[:40000], Y_train[:40000], validation_data=(x_test, y_test), epochs=epochs, batch_size=100)
# Final evaluation of the model
scores = model3.evaluate(X_train, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

########################
########################
