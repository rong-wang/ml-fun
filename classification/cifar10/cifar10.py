import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.datasets import cifar10
import os
import matplotlib.pyplot as plt
import numpy as np

def augment_image(img):
  img = tf.image.flip_left_right(img)
  img = tf.image.rgb_to_grayscale(img)
  return img

checkpoint_path = "cifar10/training1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True)

(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

train_images = train_images / 255.0
test_images = test_images / 255.0

train_labels = keras.utils.to_categorical(train_labels, num_classes=10)
test_labels = keras.utils.to_categorical(test_labels, num_classes=10)

augmented_train_images = np.zeros(shape=train_images.shape)
augmented_test_images = np.zeros(shape=test_images.shape)

i = 0
for img in train_images:
  new_img = augment_image(img)
  augmented_train_images[i] = new_img
  i += 1

i = 0
for img in test_images:
  new_img = augment_image(img)
  augmented_test_images[i] = new_img
  i += 1

new_train = np.concatenate((train_images, augmented_train_images))
new_train_labels = np.concatenate((train_labels, train_labels))

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

resnet = keras.applications.ResNet50(include_top=False, weights='imagenet', input_shape=(32,32,3))

x = resnet.output
x = keras.layers.GlobalAveragePooling2D() (x)
x = keras.layers.Dropout(0.7)(x)
predictions = keras.layers.Dense(10, activation='softmax') (x)

model = tf.keras.Model(inputs=resnet.input, outputs=predictions)

model.compile(optimizer='adam',
              loss=tf.keras.losses.categorical_crossentropy,
              metrics=['accuracy'],
              )


# model.fit(train_images, train_labels, epochs=30, callbacks=[cp_callback])
model.fit(new_train, new_train_labels, epochs=15, callbacks=[cp_callback])

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print('\nTest accuracy: ', test_acc)

model.save("cifar10/resnet.hd5")

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print('\nTest accuracy: ', test_acc)

print(test_images[0].shape)

img = test_images[0].reshape((1, 32, 32, 3))
print(class_names[np.argmax(test_labels[0])])

predictions = model.predict(img)

plt.imshow(img)
plt.show()

print("I'm ", np.max(predictions) * 100 , " percent sure this is a ", class_names[np.argmax(predictions)])