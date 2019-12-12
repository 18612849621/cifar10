import tensorflow as tf


def read(batchsize=64, type=1, no_aug_data=1):
    # Creat TFRecordReader
    reader = tf.TFRecordReader()
    if type == 0:  # train
        file_list = ['data/train.tfrecord']
    if type == 1:  # test
        file_list = ['data/test.tfrecord']
    # Set file queue
    filename_queue = tf.train.string_input_producer(
        file_list, num_epochs=None, shuffle=True
    )
    # Extract the data from TFRecord with TFRecordReader
    _, serialized_example = reader.read(filename_queue)
    # Set batch parameter
    batch = tf.train.shuffle_batch([serialized_example], batchsize,
                                   capacity=batchsize * 10,
                                   min_after_dequeue=batchsize * 5)
    # Set read data type
    feature = {
        'image': tf.FixedLenFeature([], tf.string),
        'label': tf.FixedLenFeature([], tf.int64)
    }
    # Read data
    features = tf.parse_example(batch, features=feature)
    # Extract pictures in tfrecord
    images = features['image']
    # Image data transcoding and reshape
    img_batch = tf.decode_raw(images, tf.uint8)
    img_batch = tf.reshape(img_batch, [batchsize, 32, 32, 3])
    # Picture data enhancement
    if type == 0 and no_aug_data == 1:
        # Random crop
        distorted_image = tf.random_crop(img_batch,
                                         [batchsize, 28, 28, 3])
        distorted_image = tf.image.random_contrast(distorted_image,
                                                   lower=0.5,
                                                   upper=1.5)
        distorted_image = tf.image.random_hue(distorted_image,
                                              max_delta=0.2)
        distorted_image = tf.image.random_saturation(distorted_image,
                                                     lower=0.5,
                                                     upper=1.5)
        img_batch = tf.clip_by_value(distorted_image, 0, 255)  # Range constraint
    # Reshape the size of image
    img_batch = tf.image.resize_images(img_batch, [32, 32])
    # Conversion the img_batch and label_batch data types
    label_batch = tf.cast(features['label'], tf.int64)
    # Normalized
    # Orginial picture data-range: 0 ~ 255
    # /128: 0 ~ 2
    # -1: -1 ~ 1
    img_batch = tf.cast(img_batch, tf.float32) / 128.0 - 1.0
    return img_batch, label_batch