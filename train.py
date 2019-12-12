import tensorflow as tf
import readcifar10
import resnet
import os

slim = tf.contrib.slim


def model(image, keep_prob=0.8, is_training=True):
    # Define the parameter of batch-norm
    batch_norm_params = {
        "is_training": True,  # Train:True Test:False
        "epsilon": 1e-5,  # Prevent division by zero due to normalization
        "decay": 0.997,  # Attenuation coefficient
        "scale": True,
        "updates_collections": tf.GraphKeys.UPDATE_OPS
    }
    with slim.arg_scope(
        [slim.conv2d],
        # Initialize convolution parameters in a way that
        # the variance scale is unchanged(方差尺度不变)
        weights_initializer=slim.variance_scaling_initializer(),
        # Define active Function
        activation_fn=tf.nn.relu,
        # Define weight_regularizer way L2
        weights_regularizer=slim.l2_regularizer(0.0001),
        # Define batch-norm
        normalizer_fn=slim.batch_norm,
        normalizer_params=batch_norm_params
    ):
        # Add constraint on max pooling layer
        with slim.arg_scope([slim.max_pool2d], padding='SAME'):
            # Define the first convolution
            # (input, channels, size, name) others Predefined
            net = slim.conv2d(image, 32, [3, 3], scope='conv1')
            net = slim.conv2d(net, 32, [3, 3], scope='conv2')
            # Equivalent to double downsampling
            net = slim.max_pool2d(net, [3, 3], stride=2, scope="pool1")
            # Equivalent to four times downsampling
            net = slim.conv2d(net, 64, [3, 3], scope='conv3')
            net = slim.conv2d(net, 64, [3, 3], scope='conv4')
            net = slim.max_pool2d(net, [3, 3], stride=2, scope="pool2")
            # Equivalent to eight times downsampling
            net = slim.conv2d(net, 128, [3, 3], scope='conv5')
            net = slim.conv2d(net, 128, [3, 3], scope='conv6')
            net = slim.max_pool2d(net, [3, 3], stride=2, scope="pool3")

            net = slim.conv2d(net, 256, [3, 3], scope='conv7')
            tf.reduce_mean(net, axis=[1, 2])  # nhwc --> n11c
            net = slim.flatten(net)  # Flat net n11c--> c dim vec
            # Extracted features
            net = slim.fully_connected(net, 1024)
            # dropout layer (Train-keep_prob < 1 Test-keep_prob = 1)
            slim.dropout(net, keep_prob)
            net = slim.fully_connected(net, 10)
    return net  # 10 dim vec


def loss(logist, label):
    # One hot encoding for label
    one_hot_label = slim.one_hot_encoding(label, 10)
    # Calculate cross entropy
    slim.losses.softmax_cross_entropy(logist, one_hot_label)

    # Get the REGULARIZATION_LOSSES with tf.get_collection
    reg_set = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    # Calculate the l2_loss
    l2_loss = tf.add_n(reg_set)
    # Add l2_loss
    slim.losses.add_loss(l2_loss)

    # Get the total loss
    total_loss = slim.losses.get_total_loss()

    return total_loss, l2_loss


def func_optimal(batchsize, loss_val):
    global_step = tf.Variable(0, trainable=False)
    # Set learning rate
    # 0.001: start learning rate
    # decay_step: Decay speed
    # decay_rate: Attenuation coefficient
    # staircase: decide if it's a step down False: Smooth curve
    lr = tf.train.exponential_decay(0.001,
                                    global_step,
                                    decay_steps=50000//batchsize,
                                    decay_rate=0.9,
                                    staircase=False)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    # Update the BN layer
    with tf.control_dependencies(update_ops):
        op = tf.train.AdamOptimizer(lr).minimize(loss_val, global_step)
    return global_step, op, lr


def train():
    batchsize = 64
    floder_log = 'logdirs-resnet'
    floder_model = 'model-resnet'
    # To store test information for training test samples
    tr_summary = set()
    te_summary = set()
    if not os.path.exists(floder_log):
        os.mkdir(floder_log)
    if not os.path.exists(floder_model):
        os.mkdir(floder_model)
    # data
    tr_im, tr_label = readcifar10.read(batchsize, 0, 1)
    te_im, te_label = readcifar10.read(batchsize, 1, 0)
    # net
    input_data = tf.placeholder(tf.float32, shape=[None, 32, 32, 3],
                                name='input_data')
    input_label = tf.placeholder(tf.int64, shape=[None],
                                 name='input_label')
    keep_prob = tf.placeholder(tf.float32, shape=None,
                               name='keep_prob')
    is_training = tf.placeholder(tf.bool, shape=None,
                                 name='is_training')
    logits = resnet.model_resnet(input_data, keep_prob=keep_prob, is_training=is_training)
    # loss
    total_loss, l2_loss = loss(logits, input_label)
    tr_summary.add(tf.summary.scalar('train_total_loss', total_loss))
    tr_summary.add(tf.summary.scalar('train_l2_loss', l2_loss))

    te_summary.add(tf.summary.scalar('test_total_loss', total_loss))
    te_summary.add(tf.summary.scalar('test_l2_loss', l2_loss))
    # accuracy
    pred_max = tf.argmax(logits, 1)
    correct = tf.equal(pred_max, input_label)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    tr_summary.add(tf.summary.scalar('train_accuracy', accuracy))
    te_summary.add(tf.summary.scalar('test_accuracy', accuracy))
    # op
    global_step, op, lr = func_optimal(batchsize, total_loss)
    tr_summary.add(tf.summary.scalar('train_lr', lr))
    te_summary.add(tf.summary.scalar('test_lr', lr))

    tr_summary.add(tf.summary.image('train_image', input_data * 128 + 128))
    te_summary.add(tf.summary.image('test_image', input_data * 128 + 128))

    with tf.Session() as sess:
        sess.run(tf.group(tf.global_variables_initializer(),
                          tf.local_variables_initializer()))
        tf.train.start_queue_runners(sess=sess,
                                     coord=tf.train.Coordinator())
        # Save model
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

        ckpt = tf.train.latest_checkpoint(floder_model)

        if ckpt:
            saver.restore(sess, ckpt)

        epoch_val = 100

        tr_summary_op = tf.summary.merge(list(tr_summary))
        te_summary_op = tf.summary.merge(list(te_summary))

        summary_writer = tf.summary.FileWriter(floder_log, sess.graph)
        for i in range(50000 // batchsize * epoch_val):
            train_im_batch, train_label_batch = sess.run([tr_im, tr_label])
            feed_dict = {
                input_data: train_im_batch,
                input_label: train_label_batch,
                keep_prob: 0.8,
                is_training: True
            }

            _, \
            global_step_val, \
            lr_val, \
            total_loss_val, \
            accuracy_val, \
            tr_summary_str = sess.run([op,
                                      global_step,
                                      lr,
                                      total_loss,
                                      accuracy, tr_summary_op],
                                      feed_dict=feed_dict)
            summary_writer.add_summary(tr_summary_str, global_step_val)

            if i % 100 == 0:
                print("{},{},{},{}".format(global_step_val,
                                           lr_val,
                                           total_loss_val,
                                           accuracy_val))
            if i % 100 == 0:  # On test
                test_loss = 0
                test_acc = 0
                for ii in range(1):
                    test_im_batch, test_label_batch = \
                        sess.run([te_im, te_label])
                    feed_dict = {
                        input_data: test_im_batch,
                        input_label: test_label_batch,
                        keep_prob: 1.0,
                        is_training: False
                    }
                    one_batch_loss_val, \
                    one_batch_accuracy_val, \
                    global_step_val, \
                    te_summary_str = sess.run([total_loss,
                                              accuracy,
                                              global_step,
                                              te_summary_op],
                                              feed_dict=feed_dict)

                    summary_writer.add_summary(te_summary_str, global_step_val)
                    test_loss += one_batch_loss_val
                    test_acc += one_batch_accuracy_val
                print("test:", test_loss, test_acc)
            if i % 1000:
                saver.save(sess, "{}/model.ckpt".format(floder_model, str(global_step_val)))
            # checkpoint: Documented latest model
            # meta: Define the structure of Graph
            # data: Store the value of a variable
            # index: Index for data and meta
    return


if __name__ == '__main__':
    train()


