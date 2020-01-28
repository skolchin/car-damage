#-------------------------------------------------------------------------------
# Name:        U-Net DNN testing script
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
import json
import skimage.draw
import tensorflow as tf

from matplotlib import pyplot as plt
from pathlib import Path

BATCH_SIZE = 32
EPOCHS = 20
IMG_WIDTH = 128
IMG_HEIGHT = 128
MODEL_NAME = "u-net-" + str(IMG_WIDTH)

# Display a sample image
# Source: https://www.tensorflow.org/tutorials/images/segmentation
# Author: Tensorflow Authors
def display_image(display_list):
    display_list = [x for x in display_list if x is not None]

    plt.figure(figsize=(8, 8))
    title = ['Input Image', 'True Mask', 'Predicted Mask']
    for n, x in enumerate(display_list):
        plt.subplot(1, len(display_list), n+1)
        plt.title(title[n])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(x))
        plt.axis('off')
    plt.show()

# Convert RGB color to RGBA
def rgb_to_rgba(rgb):
    row, col, ch = rgb.shape

    if ch == 4:
        return rgb
    if ch < 3:
        raise ValueError("Only 3-channel images can be converted to RGBA")

    rgba = np.ones( (row, col, 4), dtype=rgb.dtype )
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]

    rgba[:,:,0] = r
    rgba[:,:,1] = g
    rgba[:,:,2] = b

    return rgba

# Display image with mask overlayed
def display_image_with_mask(display_list, title = None):
    display_list = [x for x in display_list if x is not None]

    plt.figure(figsize=(8, 8))
    if title is not None:
        plt.title(title)

    img = display_list[0]
    plt.subplot(1, len(display_list), 1)

    plt.imshow(img)
    plt.axis('off')

    for n, mask in enumerate(display_list[1:]):
        plt.subplot(1, len(display_list), n+2)

        masked_img = rgb_to_rgba(img.numpy() if hasattr(img, "numpy") else img)
        idx = np.squeeze(mask) > 0
        masked_img[idx, 0] += 0.3
        masked_img[idx, 3] = 0.5

        plt.imshow(masked_img)
        plt.axis('off')

    plt.show()

# Display training history
# Source: https://www.tensorflow.org/tutorials/images/segmentation
# Author: Tensorflow Authors
def display_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = len(acc)
    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

# Convert prediction to mask
# Source: https://www.tensorflow.org/tutorials/images/segmentation
# Author: Tensorflow Authors
def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]

# Display prediction results on a dataset
# Source: https://www.tensorflow.org/tutorials/images/segmentation
# Author: Tensorflow Authors
def display_prediction(model, dataset, num=1):
    for num, x in enumerate(dataset.take(num)):
        image, mask = x
        pred_mask = model.predict(image)
        display_image_with_mask([image[0], mask[0], create_mask(pred_mask)],
                                "Image {}".format(num+1))

# Get U-net model based on Keras MobileNet V2
# Source: https://www.tensorflow.org/tutorials/images/segmentation
# Author: Tensorflow Authors
def get_unet_on_mobilnet():

    def upsample(filters, size, apply_dropout=False):
      initializer = tf.random_normal_initializer(0., 0.02)

      result = tf.keras.Sequential()
      result.add(
        tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                        padding='same',
                                        kernel_initializer=initializer,
                                        use_bias=False))

      result.add(tf.keras.layers.BatchNormalization())

      if apply_dropout:
          result.add(tf.keras.layers.Dropout(0.5))

      result.add(tf.keras.layers.ReLU())
      return result

    # Downsampling stack
    base_model = tf.keras.applications.MobileNetV2(input_shape=[IMG_HEIGHT, IMG_WIDTH, 3], include_top=False)

    # Use the activations of these layers
    layer_names = [
        'block_1_expand_relu',   # 64x64
        'block_3_expand_relu',   # 32x32
        'block_6_expand_relu',   # 16x16
        'block_13_expand_relu',  # 8x8
        'block_16_project',      # 4x4
    ]
    layers = [base_model.get_layer(name).output for name in layer_names]

    # Create the feature extraction model
    down_stack = tf.keras.Model(inputs=base_model.input, outputs=layers)
    down_stack.trainable = False

    # Upsampling stack
    up_stack = [
        upsample(512, 3),  # 4x4 -> 8x8
        upsample(256, 3),  # 8x8 -> 16x16
        upsample(128, 3),  # 16x16 -> 32x32
        upsample(64, 3),   # 32x32 -> 64x64
    ]

    # This is the last layer of the model
    last = tf.keras.layers.Conv2DTranspose(
      3, 3, strides=2,
      padding='same', activation='softmax')  #64x64 -> 128x128

    inputs = tf.keras.layers.Input(shape=input_shape)
    x = inputs

    # Downsampling through the model
    skips = down_stack(x)
    x = skips[-1]
    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        concat = tf.keras.layers.Concatenate()
        x = concat([x, skip])

    x = last(x)
    return tf.keras.Model(inputs=inputs, outputs=x)

# Get hands-on ("clean") U-net model
# Source: https://www.depends-on-the-definition.com/unet-keras-segmenting-images/
# Author: Tobias Sterbak
def get_unet_clean(im_shape, n_filters=16, dropout=0.5, batchnorm=True):

    def conv2d_block(input_tensor, n_filters, kernel_size=3, batchnorm=True):
        # first layer
        x = tf.keras.layers.Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
                   padding="same")(input_tensor)
        if batchnorm:
            x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)

        # second layer
        x = tf.keras.layers.Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
                   padding="same")(x)
        if batchnorm:
            x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)
        return x

    # Input
    input_img = tf.keras.layers.Input(im_shape, name='img')

    # contracting path
    c1 = conv2d_block(input_img, n_filters=n_filters*1, kernel_size=3, batchnorm=batchnorm)
    p1 = tf.keras.layers.MaxPooling2D((2, 2)) (c1)
    p1 = tf.keras.layers.Dropout(dropout*0.5)(p1)

    c2 = conv2d_block(p1, n_filters=n_filters*2, kernel_size=3, batchnorm=batchnorm)
    p2 = tf.keras.layers.MaxPooling2D((2, 2)) (c2)
    p2 = tf.keras.layers.Dropout(dropout)(p2)

    c3 = conv2d_block(p2, n_filters=n_filters*4, kernel_size=3, batchnorm=batchnorm)
    p3 = tf.keras.layers.MaxPooling2D((2, 2)) (c3)
    p3 = tf.keras.layers.Dropout(dropout)(p3)

    c4 = conv2d_block(p3, n_filters=n_filters*8, kernel_size=3, batchnorm=batchnorm)
    p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2)) (c4)
    p4 = tf.keras.layers.Dropout(dropout)(p4)

    c5 = conv2d_block(p4, n_filters=n_filters*16, kernel_size=3, batchnorm=batchnorm)

    # expansive path
    u6 = tf.keras.layers.Conv2DTranspose(n_filters*8, (3, 3), strides=(2, 2), padding='same')(c5)
    u6 = tf.keras.layers.concatenate([u6, c4])
    u6 = tf.keras.layers.Dropout(dropout)(u6)
    c6 = conv2d_block(u6, n_filters=n_filters*8, kernel_size=3, batchnorm=batchnorm)

    u7 = tf.keras.layers.Conv2DTranspose(n_filters*4, (3, 3), strides=(2, 2), padding='same')(c6)
    u7 = tf.keras.layers.concatenate([u7, c3])
    u7 = tf.keras.layers.Dropout(dropout)(u7)
    c7 = conv2d_block(u7, n_filters=n_filters*4, kernel_size=3, batchnorm=batchnorm)

    u8 = tf.keras.layers.Conv2DTranspose(n_filters*2, (3, 3), strides=(2, 2), padding='same') (c7)
    u8 = tf.keras.layers.concatenate([u8, c2])
    u8 = tf.keras.layers.Dropout(dropout)(u8)
    c8 = conv2d_block(u8, n_filters=n_filters*2, kernel_size=3, batchnorm=batchnorm)

    u9 = tf.keras.layers.Conv2DTranspose(n_filters*1, (3, 3), strides=(2, 2), padding='same') (c8)
    u9 = tf.keras.layers.concatenate([u9, c1], axis=3)
    u9 = tf.keras.layers.Dropout(dropout)(u9)
    c9 = conv2d_block(u9, n_filters=n_filters*1, kernel_size=3, batchnorm=batchnorm)

    outputs = tf.keras.layers.Conv2DTranspose(3, (3, 3), strides=1,
      padding='same', activation='softmax') (c9)

    model = tf.keras.models.Model(inputs=[input_img], outputs=[outputs])
    return model

# Load annotations of a dataset split
def load_annotations(split):
    root = Path(".\\custom").joinpath(split)
    file_name = str(root.joinpath("via_region_data.json"))

    with open(file_name,'r',encoding="utf8",errors='ignore') as f:
        anno_json = json.load(f)
        f.close()

    files, annotations = [], []
    for a in anno_json.values():
        if not 'regions' in a:
            continue

        f = root.joinpath(a['filename'])
        ##print('Split {}: adding file {}'.format(split, str(f)))
        if not f.exists():
            print("Cannot find file", f)
        else:
            polygons = []
            for r in a['regions'].values():
                all_x = [x for x in r['shape_attributes']['all_points_x']]
                all_y = [y for y in r['shape_attributes']['all_points_y']]
                polygons.extend([all_x, all_y])

            files.extend([str(f)])
            annotations.extend([polygons])

    return files, annotations

# Transform file and polygon arrays to image data and mask
def map_fn(path, polygon):
    # Load the image
    image = tf.image.decode_image(tf.io.read_file(path), channels=3, dtype=tf.float32)

    # Create mask
    mask = np.zeros((image.shape[0], image.shape[1], 1), dtype=np.uint8)

    # Polygon area (class 2)
    rr, cc = skimage.draw.polygon(polygon[1], polygon[0], image.shape)
    mask[rr, cc] = 2

    # Polygon perimeter (class 1)
    rr, cc = skimage.draw.polygon_perimeter(polygon[1], polygon[0], image.shape)
    mask[rr, cc] = 1

    image = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH])
    mask = tf.image.resize(mask, [IMG_HEIGHT, IMG_WIDTH])

    return image, mask

# Get dataset split
def get_dataset(split, show_samples=True):
    # Load annotations and polygons
    files, areas = load_annotations(split)

    # Convert to image data and mask
    file_t, area_t = [], []
    for n, x in enumerate(zip(files, areas)):
        img, mask = map_fn(x[0], x[1])
        if show_samples and n < 3:
            display_image_with_mask([img, mask])
        file_t.extend([img])
        area_t.extend([mask])
    print("{} files found in split'{}'".format(len(file_t), split))

    # Convert to tensors
    file_t = tf.convert_to_tensor(file_t)
    area_t = tf.convert_to_tensor(area_t)

    # Make a dataset
    dataset = tf.data.Dataset.from_tensor_slices((file_t, area_t))
    dataset = dataset.repeat().shuffle(100, reshuffle_each_iteration=False)
    dataset = dataset.batch(BATCH_SIZE)

    return dataset, file_t.shape[0]

# Load data
print("==> Loading dataset")
train_dataset, train_count = get_dataset('train', False)
val_dataset, val_count = get_dataset('val', False)

# Build or load the model
model_dir = ".\\models\\" + MODEL_NAME
if Path(model_dir).exists():
    print("==> Loading model", model_dir)
    model = tf.keras.models.load_model(model_dir)
else:
    print("==> Bulding model", model_dir)
    model = get_unet_clean(im_shape=[IMG_HEIGHT, IMG_WIDTH, 3], n_filters=16,
                           dropout=0.05, batchnorm=True)

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()

    #tf.keras.utils.plot_model(model, show_shapes=True)

    print("==> Training model", model_dir)
    callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir="_logs",
                                       histogram_freq=1,
                                       write_graph=True,
                                       write_grads=True,
                                       write_images=True,
                                       update_freq="batch")
    ]

    # Train the model
    model_history = model.fit(train_dataset,
                              epochs=EPOCHS,
                              steps_per_epoch=train_count,
                              validation_steps=val_count,
                              validation_data=val_dataset,
                              callbacks=callbacks)

    print("==> Saving model", model_dir)
    model.save(model_dir)
    display_history(model_history)

print("==> Prediction")

# Show predictions for dataset images
#display_prediction(model, val_dataset.shuffle(100), 5)

# Show predictions for stored images
images_dir = Path('.\\custom\\test')
image_paths = sorted(list(images_dir.glob("*.jpg")))

for image_path in image_paths:
    print(image_path)
    img = tf.image.decode_image(tf.io.read_file(str(image_path)), channels=3, dtype=tf.float32)
    img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])

    pred_mask = model.predict(img[tf.newaxis, ...])
    display_image_with_mask([img, None, create_mask(pred_mask)], str(image_path))
