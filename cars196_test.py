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

def display_samples(dataset, class_names):
    n = 1
    plt.figure(figsize=(10,10))

    for t, x in dataset.enumerate():
        if n > 25: break
        bbox, image, label = x['bbox'], x['image'], x['label']

        ax = plt.subplot(5,5,n)
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

        n += 1

    plt.tight_layout()
    plt.show()

def display_batch_samples(dataset, class_names):
    n = 1
    for t, x in dataset.enumerate():
        if n > 25: break
        images, labels = x['image'], x['label']
        plt.figure(figsize=(5,5))
        plt.axis('off')
        for image, label in zip(images, labels):
            if n > 25: break
            ax = plt.subplot(5,5,n)
            plt.title(
                label = textwrap.fill(class_names[label.numpy()], 20),
                fontdict = {'verticalalignment': 'center'}
            )
            plt.title(str(label.numpy()))
            n += 1

    plt.tight_layout()
    plt.show()


builder = tfds.builder('cars196')
dataset = builder.as_dataset(shuffle_files = True, split = tfds.Split.TEST)
display_samples(dataset, builder.info.features['label'].names)

#dataset = builder.as_dataset(batch_size = 32, shuffle_files = True, split = tfds.Split.TEST)
#display_batch_samples(dataset)

#tfds.show_examples(builder.info, dataset)

