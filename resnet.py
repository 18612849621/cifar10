import tensorflow as tf
slim = tf.contrib.slim
# net : Feature map
# num_out : Num of out channel
# down : 1 * 1 convolution kernel down-sample ratio
# stride : Decide whether to down-sample
# is_training : the BN of parameter


def resnet_blockneck(net, num_out, down, stride, is_training):
    batch_norm_params = {
        'is_training': is_training,
        'decay': 0.997,
        'epsilon': 1e-5,
        'scale': True,
        'updates_collections': tf.GraphKeys.UPDATE_OPS
    }
    with slim.arg_scope(
                [slim.conv2d],
                weights_regularizer=slim.l2_regularizer(0.0001),
                weights_initializer=slim.variance_scaling_initializer(),
                activation_fn=tf.nn.relu,
                normalizer_fn=slim.batch_norm,
                normalizer_params=batch_norm_params):
        with slim.arg_scope([slim.batch_norm], **batch_norm_params):
            with slim.arg_scope([slim.conv2d, slim.max_pool2d], padding='SAME') as arg_sc:
                # Back up the feature map first, ready for shortcut
                shortcut = net
                # Performing convolution and pooling judgments
                if num_out != net.get_shape().as_list()[-1]:
                    shortcut = slim.conv2d(net, num_out, [1, 1])
                if stride != 1:
                    shortcut = slim.max_pool2d(shortcut, [3, 3],
                                               stride=stride)
                net = slim.conv2d(net, num_out // down, [1, 1])
                net = slim.conv2d(net, num_out // down, [3, 3])
                net = slim.conv2d(net, num_out, [1, 1])

                if stride != 1:
                    net = slim.max_pool2d(net, [3, 3],
                                          stride=stride)

                net += shortcut

                return net


def model_resnet(net, keep_prob=0.5, is_training=True):
    with slim.arg_scope([slim.conv2d, slim.max_pool2d], padding='SAME') as arg_sc:
        # At the beginning, the input image is usually convolution
        net = slim.conv2d(net, 64, [3, 3], activation_fn=tf.nn.relu)
        net = slim.conv2d(net, 64, [3, 3], activation_fn=tf.nn.relu)

        net = resnet_blockneck(net, 128, 4, 2, is_training)
        net = resnet_blockneck(net, 128, 4, 1, is_training)
        net = resnet_blockneck(net, 256, 4, 2, is_training)
        net = resnet_blockneck(net, 256, 4, 1, is_training)
        net = resnet_blockneck(net, 512, 4, 2, is_training)
        net = resnet_blockneck(net, 512, 4, 1, is_training)

        net = tf.reduce_mean(net, [1, 2])  # nhwc --> n11c
        net = slim.flatten(net)  # Flat net n11c--> c dim vec

        net = slim.fully_connected(net, 1024, activation_fn=tf.nn.relu, scope='fc1')
        net = slim.dropout(net, keep_prob, scope='dropout1')
        net = slim.fully_connected(net, 10, activation_fn=None, scope='fc2')

    return net

