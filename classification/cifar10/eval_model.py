import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
import random
import numpy as np
import math

'''
EvalModel - class that implements the same methods lower, but just loads up the cifar10 data
'''
class EvalModel:
    def __init__(self, model_path):
        self.model = keras.models.load_model(model_path)
        (train_images, train_labels), (test_images, test_labels) = keras.datasets.cifar10.load_data()
        train_images = train_images / 255.
        test_images = test_images / 255.

        train_labels = keras.utils.to_categorical(train_labels, num_classes=10)
        test_labels = keras.utils.to_categorical(test_labels, num_classes=10)

        self.train_images = train_images
        self.train_labels = train_labels
        self.test_images = test_images
        self.test_labels = test_labels

        self.class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    def eval(self):
        loss, acc = self.model.evaluate(self.test_images, self.test_labels)
        print("Model has ", acc * 100, "accuracy")

    def sample_eval_model(self, num_images=20):
        plt_dimension = math.ceil(math.sqrt(num_images))

        images_indexes = random.sample(range(self.test_images.shape[0]), num_images)

        plt.figure(figsize=(20, 20))
        plt.subplots_adjust(hspace=0.5)

        right = 0
        wrong = 0

        for i in range(num_images):
            plt.subplot(plt_dimension, plt_dimension, i + 1)
            img = self.test_images[images_indexes[i]]
            img_reshape = img.reshape((1, 32, 32, 3))
            prediction = (self.model.predict(img_reshape))
            plt.imshow(img)
            plt.axis('off')

            label = self.class_names[self.test_labels[images_indexes[i]][0]]
            predict_str = self.class_names[np.argmax(prediction)]

            if label == predict_str:
                right += 1
                plt.setp(plt.title(label + "==" + predict_str), color='green')
            else:
                wrong += 1
                plt.setp(plt.title(label + "!=" + predict_str), color='red')

        plt.show()

        print("Right: ", right)
        print("Wrong: ", wrong)
        print("Accuracy: ", right / num_images)

    def get_confusion_matrix(self):
        predictions = self.model.predict(self.test_images)
        predictions_processed = [None] * predictions.shape[0]

        i = 0
        for vec in predictions:
            predictions_processed[i] = np.argmax(vec)
            i += 1

        tform_matrix = tf.math.confusion_matrix(labels=self.test_labels, predictions=predictions_processed, num_classes=10)
        return tform_matrix

    def load_new_model(self, new_model_path):
        self.model = keras.models.load_model(new_model_path)

def eval_model(model_path, num_images=20):
    plt_dimension = math.ceil(math.sqrt(num_images))

    model = keras.models.load_model(model_path)

    (train_images, train_labels), (test_images, test_labels) = keras.datasets.cifar10.load_data()
    train_images = train_images / 255
    test_images = test_images / 255

    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    shape = test_images.shape
    images_indexes = random.sample(range(shape[0]), num_images)

    plt.figure(figsize=(20, 20))
    plt.subplots_adjust(hspace=0.5)

    right = 0
    wrong = 0

    for i in range(num_images):
        plt.subplot(plt_dimension, plt_dimension, i + 1)
        img = test_images[images_indexes[i]]
        img_reshape = img.reshape((1, 32, 32, 3))
        prediction = (model.predict(img_reshape))
        plt.imshow(img)
        plt.axis('off')

        label = class_names[test_labels[images_indexes[i]][0]]
        predict_str = class_names[np.argmax(prediction)]

        title = 0
        if label == predict_str:
            right += 1
            title = plt.title(label + "==" + predict_str)
            plt.setp(title, color='green')
        else:
            wrong += 1
            title = plt.title(label + "!=" + predict_str)
            plt.setp(title, color='red')

    plt.show()

    print("Right: ", right)
    print("Wrong: ", wrong)
    print("Accuracy: ", right / num_images)

def get_confusion_matrix(model_path):
    model_path = "resnet1"
    model = keras.models.load_model(model_path)

    (train_images, train_labels), (test_images, test_labels) = keras.datasets.cifar10.load_data()
    train_images = train_images / 255.
    test_images = test_images / 255.

    predictions = model.predict(test_images)
    predictions_processed = [None] * predictions.shape[0]

    i = 0
    for p in predictions:
        predictions_processed[i] = np.argmax(p)
        i += 1

    tform_matrix = tf.math.confusion_matrix(labels=test_labels, predictions=predictions_processed, num_classes=10)
    return tform_matrix

model = EvalModel("resnet.h5")

model.eval()

# matrix = model.get_confusion_matrix()
# print(matrix)