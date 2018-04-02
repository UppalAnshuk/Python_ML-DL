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

A simple CNN trained on the famous MNIST dataset, details are as below-
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
None
Train on 40000 samples, validate on 10000 samples
Epoch 1/20
40000/40000 [==============================] - 634s 16ms/step - loss: 0.2594 - acc: 0.9211 - val_loss: 0.0504 - val_acc: 0.9843
Epoch 2/20
40000/40000 [==============================] - 624s 16ms/step - loss: 0.0878 - acc: 0.9736 - val_loss: 0.0391 - val_acc: 0.9866
Epoch 3/20
40000/40000 [==============================] - 649s 16ms/step - loss: 0.0650 - acc: 0.9808 - val_loss: 0.0335 - val_acc: 0.9882
Epoch 4/20
40000/40000 [==============================] - 628s 16ms/step - loss: 0.0512 - acc: 0.9843 - val_loss: 0.0262 - val_acc: 0.9907
Epoch 5/20
40000/40000 [==============================] - 622s 16ms/step - loss: 0.0452 - acc: 0.9866 - val_loss: 0.0250 - val_acc: 0.9907
Epoch 6/20
40000/40000 [==============================] - 627s 16ms/step - loss: 0.0381 - acc: 0.9888 - val_loss: 0.0247 - val_acc: 0.9913
Epoch 7/20
40000/40000 [==============================] - 640s 16ms/step - loss: 0.0347 - acc: 0.9895 - val_loss: 0.0233 - val_acc: 0.9919
Epoch 8/20
40000/40000 [==============================] - 752s 19ms/step - loss: 0.0316 - acc: 0.9904 - val_loss: 0.0218 - val_acc: 0.9919
Epoch 9/20
40000/40000 [==============================] - 782s 20ms/step - loss: 0.0296 - acc: 0.9913 - val_loss: 0.0222 - val_acc: 0.9921
Epoch 10/20
40000/40000 [==============================] - 845s 21ms/step - loss: 0.0279 - acc: 0.9915 - val_loss: 0.0224 - val_acc: 0.9925
Epoch 11/20
40000/40000 [==============================] - 802s 20ms/step - loss: 0.0237 - acc: 0.9926 - val_loss: 0.0238 - val_acc: 0.9925
Epoch 12/20
40000/40000 [==============================] - 618s 15ms/step - loss: 0.0262 - acc: 0.9922 - val_loss: 0.0225 - val_acc: 0.9932
Epoch 13/20
40000/40000 [==============================] - 621s 16ms/step - loss: 0.0230 - acc: 0.9925 - val_loss: 0.0212 - val_acc: 0.9932
Epoch 14/20
40000/40000 [==============================] - 624s 16ms/step - loss: 0.0209 - acc: 0.9933 - val_loss: 0.0222 - val_acc: 0.9928
Epoch 15/20
40000/40000 [==============================] - 624s 16ms/step - loss: 0.0198 - acc: 0.9935 - val_loss: 0.0224 - val_acc: 0.9927
Epoch 16/20
40000/40000 [==============================] - 619s 15ms/step - loss: 0.0197 - acc: 0.9934 - val_loss: 0.0230 - val_acc: 0.9925
Epoch 17/20
40000/40000 [==============================] - 630s 16ms/step - loss: 0.0194 - acc: 0.9936 - val_loss: 0.0225 - val_acc: 0.9926
Epoch 18/20
40000/40000 [==============================] - 642s 16ms/step - loss: 0.0179 - acc: 0.9948 - val_loss: 0.0218 - val_acc: 0.9929
Epoch 19/20
40000/40000 [==============================] - 621s 16ms/step - loss: 0.0184 - acc: 0.9942 - val_loss: 0.0229 - val_acc: 0.9921
Epoch 20/20
40000/40000 [==============================] - 611s 15ms/step - loss: 0.0183 - acc: 0.9938 - val_loss: 0.0225 - val_acc: 0.9928
Accuracy: 99.08%
Without any fancy stuff like leaky ReLU / ELU or batch normalization this model is able to achieve very high accuracy!!
"""
