#Name- Shivani Sripadojwar
#Email- shivani@vsoftconsulting.com

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten,Conv2D,MaxPool2D,Dense,ZeroPadding2D,Dropout
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd 
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from keras.preprocessing.image import ImageDataGenerator
import cv2
from keras import backend as K


#Reading the train and test data
train_df = pd.read_csv("fashion-mnist_train.csv")
test_df = pd.read_csv("fashion-mnist_test.csv")
train_df.head()


# Create a dictionary for each type of label 
labels = {0 : "T-shirt/top", 1: "Trouser", 2: "Pullover", 3: "Dress", 4: "Coat",
          5: "Sandal", 6: "Shirt", 7: "Sneaker", 8: "Bag", 9: "Ankle Boot"}

def get_classes_distribution(data):
    # Get the count for each label
    label_counts = data["label"].value_counts()

    # Get total number of samples
    total_samples = len(data)


    # Count the number of items in each class
    for i in range(len(label_counts)):
        label = labels[label_counts.index[i]]
        count = label_counts.values[i]
        percent = (count / total_samples) * 100
        print("{:<20s}:   {} or {}%".format(label, count, percent))

get_classes_distribution(train_df)

def plot_label_per_class(data):
    f, ax = plt.subplots(1,1, figsize=(12,4))
    g = sns.countplot(data.label, order = data["label"].value_counts().index)
    g.set_title("Number of labels for each class")

    for p, label in zip(g.patches, data["label"].value_counts().index):
        g.annotate(labels[label], (p.get_x(), p.get_height()+0.1))
    plt.show()  
    
plot_label_per_class(train_df)

#splitting the data in terms of train and test 

X_train = train_df.iloc[:,1:]
Y_train = train_df.iloc[:,0]
X_test = test_df.iloc[:,1:]
Y_test = test_df.iloc[:,0]


X_train = X / 225
X_train_test = X_test / 225
X_train = X_train.values.reshape(-1,28,28,1)
X_test = X_train_test.values.reshape(-1,28,28,1)
Y_test = to_categorical(Y_test)
Y_train = to_categorical(Y)

img=X_train[30,:].reshape((28,28))
plt.imshow(img)
plt.show()


#A. image Augmentation

def imgGen(img, zca=False, rotation=0., w_shift=0., h_shift=0., shear=0., zoom=0., h_flip=False, v_flip=False,  preprocess_fcn=None, batch_size=9):
    datagen = ImageDataGenerator(
            zca_whitening=zca, # apply zca whitening
            rotation_range=rotation,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=w_shift,   # randomly shift images horizontally (fraction of total width)
            height_shift_range=h_shift,  # randomly shift images vertically (fraction of total height)
            shear_range=shear,
            zoom_range=zoom,
            fill_mode='nearest',
            horizontal_flip=h_flip,  # randomly flip images
            vertical_flip=v_flip,   # randomly flip image
            preprocessing_function=preprocess_fcn,
            data_format=K.image_data_format())
    
    datagen.fit(img)


# reshape it to prepare for data generator
img = img.astype('float32')
img /= 255
h_dim = np.shape(img)[0]
w_dim = np.shape(img)[1]
num_channel = np.shape(img)[2]
img = img.reshape(1, h_dim, w_dim, num_channel)

# generate images using function imgGen
imgGen(img, rotation=30, h_shift=0.5)

plt.figure(figsize=(10,10))
for i in range(16):
    plt.subplot(4,4,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(X_train[i], cmap=plt.cm.binary)
    plt.title("Image label is: {}".format(Y[i]))
plt.show()

np.shape(X_test)

#B. applying CNN functions stride, padding, Dropout,Dense

cnn_model = Sequential([
    ZeroPadding2D(padding=(1,1),input_shape=(28,28,1)),
    Conv2D(32,3,activation='relu'),
    Dropout(0.2),
    MaxPool2D(pool_size=2,strides=2),
    ZeroPadding2D(padding=(1,1)),
    Conv2D(64,3,activation='relu'),
    Dropout(0.2),
    MaxPool2D(pool_size=2,strides=2),
    Flatten(),
    Dense(128,activation="relu"),
    Dropout(0.5),
    Dense(10,activation="softmax")])

#summary 

cnn_model.summary()

#accuracy

cnn_model.compile("adam",loss="categorical_crossentropy",metrics=["accuracy"])


history = cnn_model.fit(X_train,Y_train,epochs=10,batch_size=128,validation_data = (X_test,Y_test))

#C. graph for accuracy and loss

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#printing loss and accuracy rate

score = cnn_model.evaluate(X_test, Y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

pred=cnn_model.predict(X_test)
print(pred)

print(Y_test)

#D. confusion Matrix
pred_classes = np.argmax(pred, axis = 1)
Y_true = np.argmax(Y_test, axis = 1)

#accuracy score

score=accuracy_score(Y_true,pred_classes)
print(score)

#F1 score, precision, recall, confusion matrix

matrix=classification_report(Y_true,pred_classes)

print("classification report \n", matrix)

