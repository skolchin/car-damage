#-------------------------------------------------------------------------------
# Name:        ImageNet additional training script
# Purpose:     Car damage analysis project
#
# Author:      kol
#
#              Parts of the code were obtained from open Internet sources
#              Source and authors of such are listed below
#
# Created:     27.01.2020
# Copyright:   (c) kol 2020
# Licence:     MIT
#-------------------------------------------------------------------------------

import numpy as np
import pathlib
import matplotlib
import itertools

from matplotlib import pyplot as plt

import tensorflow as tf

# Load ImageNet labels
labels_path = tf.keras.utils.get_file(
    'ImageNetLabels.txt',
    'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')
imagenet_labels = np.array(open(labels_path).read().splitlines())

# Load the model
model = tf.keras.applications.MobileNetV2()

matplotlib.use("TkAgg")

# Load and process images
images_dir = pathlib.Path('C:\\Users\\kol\\Documents\\kol\\car-damage\\custom\\test')
image_paths = sorted(list(images_dir.glob("*.jpg")))

for image_path in image_paths:
    print(image_path)
    img = tf.keras.preprocessing.image.load_img(str(image_path), target_size=[224, 224])

    x = tf.keras.preprocessing.image.img_to_array(img)
    x = tf.keras.applications.mobilenet.preprocess_input(x[tf.newaxis,...])

    result = model(x)

    labels = imagenet_labels[np.argsort(result)[0,::-1][:5] + 1]
    probs = np.sort(result)[0,::-1][:5]

    probs_pct = [np.round(x * 100, 2) for x in probs]
    label_probs = ['{}: {}%'.format(v[0], v[1]) for v in zip(labels, probs_pct)]
    print("Result:\n", label_probs)

    plt.title(image_path)
    plt.imshow(img)
    plt.text(0, 50, '\n'.join(label_probs), color='red')
    plt.axis('off')
    plt.show()
