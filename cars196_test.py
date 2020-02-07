#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      kol
#
# Created:     15.01.2020
# Copyright:   (c) kol 2020
# Licence:     <your licence>
#-------------------------------------------------------------------------------

import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import textwrap
import time

def display_samples(dataset, class_names, model=None, imagenet_labels=None):
    plt.figure(figsize=(10,10))

    for n, x in dataset.enumerate():
        n = n.numpy()
        if n >= 25: break
        bbox, image, label = x['bbox'], x['image'], x['label']

        ax = plt.subplot(5, 5, n+1)
        plt.axis('off')

        ax.imshow(image.numpy())
        plt.title(
            label = textwrap.fill(class_names[label.numpy()], 20),
            fontdict = {'verticalalignment': 'center'}
        )

        # ymin, xmin, ymax, xmax
        bbox = bbox.numpy()
        bbox[0] *= image.shape[0]
        bbox[1] *= image.shape[1]
        bbox[2] *= image.shape[0]
        bbox[3] *= image.shape[1]

        # xy, width, height
        rect = patches.Rectangle((bbox[1], bbox[0]), bbox[3]-bbox[1], bbox[2]-bbox[0],
            fill=False, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

        if model is not None:
            x = tf.cast(image, "float32") /  255.0
            x = tf.keras.applications.mobilenet.preprocess_input(x[tf.newaxis,...])

            result = model(x)

            labels = imagenet_labels[np.argsort(result)[0,::-1][:3]]
            probs = np.sort(result)[0,::-1][:3]

            probs_pct = [np.round(x * 100, 2) for x in probs]
            label_probs = ['{}: {}%'.format(v[0], v[1]) for v in zip(labels, probs_pct)]
            print(label_probs)

            plt.text(0.0, 10.0, '\n'.join(label_probs), fontsize=8, color="red")

    plt.tight_layout()
    plt.show()


# Load dataset
#builder = tfds.builder('cars196')
#dataset = builder.as_dataset(shuffle_files = True, split = tfds.Split.TEST)
#class_names = builder.info.features['label'].names
#display_samples(dataset, class_names)

# Load ImageNet labels
##labels_path = tf.keras.utils.get_file(
##    'ImageNetLabels.txt',
##    'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')
##imagenet_labels = np.array(open(labels_path).read().splitlines())
##
### Load the model
##model = tf.keras.applications.MobileNetV2(weights='imagenet')
##display_samples(dataset, class_names, model, imagenet_labels)
##

def display_dataset_samples(dataset, num_samples=25, shuffle=True):
    fig = plt.figure(figsize=(8,8))
    nrows = int(np.ceil(num_samples / 5))
    ds = dataset.shuffle(1000).take(1) if shuffle else dataset.take(1)

    for _, image_batch in ds.enumerate():
        for n, image in enumerate(image_batch):
            if n >= num_samples-1:
                break
            plt.subplot(nrows, 5, n+1)
            plt.imshow(image)
            plt.axis('off')

    plt.tight_layout()
    plt.show()

@tfds.decode.make_decoder()
def convert_range(serialized_image, feature):
    x = tf.image.decode_image(serialized_image, channels=feature.shape[-1])
    x = tf.cast(x, "float32")
    x = tf.image.resize_with_pad(x, 64, 64)
    x = (x - 127.5) / 127.5
    return x


builder = tfds.builder('cars196')
ds1 = builder.as_dataset(
                split=tfds.Split.TEST,
                batch_size=32,
                shuffle_files=True,
                #in_memory=True,
                decoders={
                    'image': convert_range()
                    })

##ds2 = builder.as_dataset(
##                split=tfds.Split.TRAIN,
##                batch_size=32,
##                shuffle_files=True,
##                in_memory=True,
##                decoders={
##                    'image': convert_range()
##                    })

ds = ds1

for _, x in ds.enumerate():
    break

def map_fn(x):
    return x['image']

ds_new = ds.map(map_fn, tf.data.experimental.AUTOTUNE)

display_dataset_samples(ds_new, shuffle=False)

