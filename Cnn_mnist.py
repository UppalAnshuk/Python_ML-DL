from sklearn.utils import shuffle
import numpy
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten
from keras.layers.convolutional import Conv2D,MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')



(a, b), (x_test, y_test) = mnist.load_data()
a,b=shuffle(a,b)
a=a.reshape(60000,1,28,28)
x_test=x_test.reshape(10000,1,28,28)
#training on the first 40000 images and evaluation on the rest
X_train=a[:40000]
Y_train=b[:40000]
x_eval=a[40000:]
y_eval=b[40000:]
X_train=X_train.astype('float32')
x_test=x_test.astype('float32')
x_eval=x_eval.astype('float32')
X_train=X_train/255.0
x_test=x_test/255.0
x_eval=x_eval/255.0

Y_train = np_utils.to_categorical(Y_train)
y_test = np_utils.to_categorical(y_test)
y_eval = np_utils.to_categorical(y_eval)
num_classes = y_test.shape[1]



model3=Sequential()
model3.add(Conv2D(20,(3,3),strides=(1,1),input_shape=(1,28,28),padding="same",data_format="channels_first",use_bias=False))
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
epochs = 20
lrate = 0.01
decay = lrate/epochs
sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
model3.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
print(model3.summary())

model3.fit(X_train, Y_train, validation_data=(x_test, y_test), epochs=epochs, batch_size=32)
# Final evaluation of the model
scores = model3.evaluate(x_eval,y_eval, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
"""

Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_1 (Conv2D)            (None, 20, 28, 28)        180       
_________________________________________________________________
dropout_1 (Dropout)          (None, 20, 28, 28)        0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 40, 22, 22)        39240     
_________________________________________________________________
dropout_2 (Dropout)          (None, 40, 22, 22)        0         
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 40, 16, 16)        78440     
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 40, 8, 8)          0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 2560)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 128)               327808    
_________________________________________________________________
dropout_3 (Dropout)          (None, 128)               0         
_________________________________________________________________
dense_2 (Dense)              (None, 10)                1290      
=================================================================
Total params: 446,958
Trainable params: 446,958
Non-trainable params: 0
_________________________________________________________________
"""
