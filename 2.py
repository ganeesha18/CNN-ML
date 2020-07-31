from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras.callbacks import EarlyStopping 
import numpy as np
import keras

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape((X_train.shape[0], 28, 28, 1)).astype('float32')
X_test = X_test.reshape((X_test.shape[0], 28, 28, 1)).astype('float32')

# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255

#generate  noice data
def noice_data(noice_factor):
    
    
    X_train_noisy=X_train + noice_factor*np.random.normal(loc=0.0, scale = 1.0, size=X_train.shape)
    X_test_noisy=X_test + noice_factor*np.random.normal(loc=0.0, scale = 1.0, size=X_test.shape)
    
    X_train_noisy = np.clip(X_train_noisy,0.,1.)
    X_test_noisy = np.clip(X_test_noisy,0.,1.) 
    return X_train_noisy,X_test_noisy

# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

def create_model():
    
    
    model = Sequential()
    model.add(Conv2D(32, (5, 5), input_shape=(28, 28, 1), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
 

callback = EarlyStopping(monitor='val_loss', patience=3)
#model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200)
for noice_fact in np.arange(0,1.050,0.050):
    print(noice_fact)
    X_train_noisy,X_test_noisy = noice_data(noice_fact)
    
    model = create_model()    
    model.fit(X_train_noisy, y_train, validation_split=0.1, epochs=1, batch_size=200,callbacks=[callback])
    
    scores = model.evaluate(X_test_noisy, y_test, verbose=0)
    print("Noice factor: %.2f%%, Accuracy: %.2f%%" % (noice_fact,scores[1]*100))


