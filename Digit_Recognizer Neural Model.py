#Author: Kelby Amandy
#Date: 4/17/2023
#Title: Doodle Classifier!

import tensorflow as tf 
from keras.datasets import mnist
import os
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np

# Loading the train and test datasets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Data augmentation
datagen = ImageDataGenerator(rotation_range=90,
                             width_shift_range=0.35,
                             height_shift_range=0.35,
                             shear_range=0.2,
                             zoom_range=0,
                             horizontal_flip=False,
                             vertical_flip=False)

#Normalizing dataset
X_train = X_train.astype('float32') / 255.0
X_val = X_val.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Reshape images to a 28x28x1 format
X_train = X_train.reshape(-1, 28, 28, 1)
X_val = X_val.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

# Define the number of augmented samples to generate
num_augmented_samples = 48000

# Generate the augmented data using the flow() method
augmented_data = datagen.flow(X_train, y_train, batch_size=num_augmented_samples, shuffle=False)

# Get the augmented images and labels as numpy arrays
augmented_X_train, augmented_y_train = augmented_data.next()

# Concatenate the original and augmented data
X_train = np.concatenate([X_train, augmented_X_train], axis=0)
y_train = np.concatenate([y_train, augmented_y_train], axis=0)

# Shuffle the data
indices = np.arange(X_train.shape[0])
np.random.shuffle(indices)
X_train = X_train[indices]
y_train = y_train[indices]


# # Showing the data
# for i in range(9):
#     # plot raw pixel data
#     plt.imshow(X_train[i], cmap=plt.get_cmap('gray'))
#     plt.title(y_train[i])
#     # show the figure
#     plt.show()


# One-hot encode labels
y_train = to_categorical(y_train)
y_val =  to_categorical(y_val)
y_test = to_categorical(y_test)

# Creating a 3 layer CNN model along with L2 Regularization and Dropout to prevent overfitting and 'same' padding to help with images with digits near edge
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1), padding='same'),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', kernel_regularizer='l2', padding='same'),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compiling the model with Adam and a learning rate of 0.001 along with categorical crossentropy for loss since it is labeled and we have 10 classes
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()


logdir = 'logs'
base_dir = os.path.join(os.path.expanduser("-"), "Documents")
logs_dir = os.path.join(base_dir, logdir)
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logs_dir)

# Training the model on 20 epochs
history = model.fit(X_train, y_train, epochs=20, validation_data=(X_val, y_val), callbacks=[tensorboard_callback], verbose=1)

#Evaluate the model on the test dataset
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', loss)
print('Test accuracy:', accuracy)

# Save the model in standard .h5 file
model.save('digit_classifier.h5')

