import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

from tensorflow.keras.datasets import mnist
(X_train,y_train) , (X_test,y_test)=mnist.load_data()

plt.imshow(X_train[0])

for i in range(20):
    plt.subplot(5, 5, i+1)
    plt.imshow(X_train[i], cmap=plt.get_cmap('gray'))

plt.show()
X_train_flat=X_train.reshape(len(X_train),28*28)
X_test_flat=X_test.reshape(len(X_test),28*28)

print(X_train_flat.shape)
print(X_test_flat.shape)
X_train_flat=X_train_flat/255
X_test_flat=X_test_flat/255

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model=Sequential()
model.add(Dense(10,input_shape=(784,),activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

model.fit(X_train_flat,y_train,epochs=10)
model.evaluate(X_test_flat,y_test)
y_predict = model.predict(X_test_flat)
