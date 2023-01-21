import numpy as np 
import pandas as pd 
import os

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
import numpy as np
import pandas as pd
import os 
from PIL import Image


os.listdir("/content/drive/MyDrive/concrete crack")
['valid', 'test', 'train', 'predict']

import matplotlib.pyplot as plt
from matplotlib import image
from PIL import Image

im = image.imread("/content/drive/MyDrive/concrete crack/train/Positive/IMG_0482_1_15.jpg")
im.shape
plt.imshow(im)

plt.imshow(im)

im = image.imread('/content/drive/MyDrive/concrete crack/train/Negative/IMG_2177_12_11.jpg')
im.shape

plt.imshow(im)

im = image.imread('/content/drive/MyDrive/concrete crack/valid/Positive/IMG_0580_11_11.jpg')
im.shape

plt.imshow(im)

im = image.imread('/content/drive/MyDrive/concrete crack/valid/Negative/IMG_0121_11_15.jpg')
im.shape

plt.imshow(im)

import numpy as np
from PIL import Image
def Load_Images(impath):
    imgs=[]
    label=[]
    l1=os.listdir(impath)
    for i in l1:
        l2 = os.listdir(impath + '/' + i)
        for j in l2:
            img = Image.open(impath + '/' + i + '/' + j)
            img = img.resize(size = (64, 64))
            img = img.convert('RGB')
            img = np.array(img, dtype = np.float16) / 255
            imgs.append(np.array(img)) 
            label.append(i)
            del img
    return np.array(imgs),label
    
x, y = Load_Images('/content/drive/MyDrive/concrete crack/train')

x.shape, len(y)

a = pd.Series(y, dtype = "category")
a

a.value_counts()

b = a.cat.codes
b

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, b, random_state = 1, test_size = 0.2)

x_train.shape, x_test.shape, y_train.shape, y_test.shape

from tensorflow.keras.layers import Conv2D, AveragePooling2D, Flatten, Dense
from tensorflow.keras.models import Sequential
model = Sequential()
model.add(Conv2D(32, kernel_size = (3,3), activation = "relu",  input_shape = x_train.shape[1:]))
model.add(AveragePooling2D())
model.add(Conv2D(64, kernel_size = (3,3), activation = "relu"))
model.add(AveragePooling2D())
model.add(Conv2D(128, kernel_size = (3,3), activation = "relu"))
model.add(AveragePooling2D())
model.add(Conv2D(64, kernel_size = (3,3), activation = "relu"))
model.add(AveragePooling2D())
model.add(Flatten())
model.add(Dense(64, activation = "relu"))
model.add(Dense(3, activation = "softmax"))
model.summary()

model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

model.evaluate(x_train, y_train)

model.evaluate(x_test,y_test)
