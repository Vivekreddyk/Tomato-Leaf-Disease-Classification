import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense,Flatten,Dropout
from keras.models import Model
from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

Image_Size = [224,224,3]
train_path = 'train'
test_path = 'test'

Vgg19 = VGG19(include_top=False, weights='imagenet', input_tensor=Image_Size)

for layer in Vgg19.layers:
    layer.trainable=False

x = Flatten()(Vgg19.output)
output = Dense(units=256, activation='relu', kernel_initializer='he_uniform' )
x = Dropout(rate=0.25)
model = Dense(units=10, activation='softmax')

print(model.summary)

train_datagen = ImageDataGenerator(
    rescale = 1./255,
    rotation_range=0.4,
    shear_range=0.2,
    zoom_range=0.2,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255,)

training_set = ImageDataGenerator.flow_from_directory(
    directory='train',
    target_size=[224,224],
    class_mode='categorical',
    batch_size=32
)

testing_set = ImageDataGenerator.flow_from_directory(
    directory='test',
    target_size=[224,224],
    class_mode='categorical',
    batch_size=32
) 

model.compile(
    loss=keras.losses.categorical_crossentropy,
    optimizer=keras.optimizers.SGD(lr=[0.0001], momentum=0.9),
    metrics = ['accuracy']
)

vgg19_model = model.fit(training_set, validation_data=testing_set, epochs=30, verbose=1, steps_per_epoch=len(training_set), validation_steps=len(testing_set))


plt.plot(r.history['accuracy'])
plt.plot(r.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Accuracy','Validation Accuracy','Loss','Validation Loss'], loc='upper right')
plt.show()

plt.plot(r.history['loss'])
plt.plot(r.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Loss','Validation Loss'], loc='upper right')
plt.show()
