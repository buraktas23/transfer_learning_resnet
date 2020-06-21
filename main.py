from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img,img_to_array
from tensorflow.keras.applications.resnet import ResNet50
import matplotlib.pyplot as plt
from glob import glob
from tensorflow.keras import backend as K

train= ".../COVID-19 Radiography Database/train/"
test=  ".../COVID-19 Radiography Database/test/"

img = load_img(train + "COVID-19/COVID-19 (1).png")
img = img_to_array(img)

input_shape = img.shape
print("input shape : ",input_shape)

num_classes = glob(test + "/*")
num_classes = len(num_classes)
print("number of classes : ",num_classes)

resnet = ResNet50(include_top=False,weights='imagenet',input_shape=(224,224,3), classes=3)

resnet.summary()


model = Sequential()

model.add(resnet)

for layers in model.layers:
  layers.trainable = False

model.add(Flatten())
model.add(Dense(1024))

model.add(Dense(num_classes,activation='softmax'))

model.summary()


model.add(Flatten())
model.add(Dense(1024))

model.add(Dense(num_classes,activation='softmax'))

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

test_data = ImageDataGenerator().flow_from_directory(test,
                                                     target_size=(224,224))

train_data = ImageDataGenerator().flow_from_directory(train,
                                                     target_size=(224,224))

model.compile(loss='categorical_crossentropy',
              optimizer = 'rmsprop',
              metrics = ['accuracy',f1_m,precision_m, recall_m])

batch_size = 32

hist = model.fit_generator(train_data,
                           steps_per_epoch = 100,
                           epochs=50,
                           validation_data=test_data,
                           validation_steps = 20)
