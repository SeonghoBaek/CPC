import tensorflow as tf
import layers
from sklearn.utils import shuffle
import numpy as np
import os
import argparse
import cv2
import util


def prepare_patches(image, patch_size=[24, 24], patch_dim=[7, 7], stride=12):
    patches = []
    patch_w = patch_size[0]
    patch_h = patch_size[1]

    #print('image shape: ', image.shape)
    for h in range(patch_dim[0]):
        for w in range(patch_dim[1]):
            #print('h:', h*stride, ' w: ', h*stride + patch_h)
            #print('w:', w*stride, ' w: ', w*stride + patch_w)
            patch = image[h*stride:(h*stride + patch_h), w*stride:(w*stride + patch_w), :]
            #print('Patch dims: ', patch.shape)
            patches.append(patch)

    #print('Num patches: ', len(patches))
    return np.array(patches)


def load_images_from_folder(folder, use_augmentation=False, add_noize=False):
    images = []

    for filename in os.listdir(folder):
        fullname = os.path.join(folder, filename).replace("\\", "/")
        jpg_img = cv2.imread(fullname)
        img = cv2.cvtColor(jpg_img, cv2.COLOR_BGR2RGB)  # To RGB format
        img = cv2.resize(img, dsize=(input_height, input_width))

        if img is not None:
            img = np.array(img)

            n_img = img / 255.0
            images.append(n_img)

            n_img = cv2.flip(img, 1)
            n_img = n_img / 255.0
            images.append(n_img)

            if use_augmentation == True:
                img = cv2.resize(img, dsize=(scale_size, scale_size), interpolation=cv2.INTER_CUBIC)

                dy = np.random.random_integers(low=1, high=img.shape[0] - input_height, size=num_aug_patch - 1)
                dx = np.random.random_integers(low=1, high=img.shape[1] - input_width, size=num_aug_patch - 1)

                window = list(zip(dy, dx))

                for i in range(len(window)):
                    croped = img[window[i][0]:window[i][0] + input_height,
                             window[i][1]:window[i][1] + input_width].copy()
                    # cv2.imwrite(filename + '_crop_' + str(i) + '.jpg', croped)
                    n_croped = croped / 255.0
                    images.append(n_croped)

                    croped = cv2.flip(croped, 1)

                    if add_noize == True:
                        croped = croped + np.random.normal(size=(input_height, input_width, num_channel))

                        croped[croped > 255.0] = 255.0
                        croped[croped < 0] = 0.0

                    croped = croped / 255.0
                    images.append(croped)

    return np.array(images)


def pixelCNN(latents, num_iteration=5, depth=256, scope='pixel_cnn'):
    cres = latents
    cres_dim = cres.shape[-1]
    num_channel = depth

    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        for _ in range(num_iteration):
            c = layers.conv(cres, scope='conv1', filter_dims=[1, 1, num_channel], stride_dims=[1, 1],
                            non_linear_fn=tf.nn.relu, bias=True)
            c = layers.conv(c, scope='conv2', filter_dims=[1, 3, num_channel], stride_dims=[1, 1],
                            non_linear_fn=None, bias=True)

            padding = tf.constant([[0, 0], [1, 0], [0, 0], [0, 0]], name='padding')
            c = tf.pad(c, padding, name='pad')

            c = layers.conv(c, scope='conv3', filter_dims=[2, 1, num_channel], stride_dims=[1, 1], padding='VALID',
                            non_linear_fn=tf.nn.relu, bias=True)

            c = layers.conv(c, scope='conv4', filter_dims=[1, 1, cres_dim], stride_dims=[1, 1],
                            non_linear_fn=None, bias=True)

            cres = cres + c

        cres = tf.nn.relu(cres)

    return cres


def CPC(latents, target_dim=64, emb_scale=0.1, steps_to_ignore=2, steps_to_predict=3, scope='cpc'):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        loss = 0.0
        context = pixelCNN(latents)
        print('PixelCNN Context Shape: ' + str(context.get_shape().as_list()))

        targets = layers.conv(latents, scope='conv1', filter_dims=[1, 1, target_dim], stride_dims=[1, 1],
                              non_linear_fn=None, bias=True)
        batch_dim, col_dim, row_dim = targets.get_shape().as_list()[:-1]
        #print(batch_dim, col_dim, row_dim)
        targets = tf.reshape(targets, [-1, target_dim])

        for i in range(steps_to_ignore, steps_to_predict):
            col_dim_i = col_dim - i - 1
            total_elements = batch_dim * col_dim_i * row_dim
            preds_i = layers.conv(context, scope='conv2', filter_dims=[1, 1, target_dim], stride_dims=[1, 1],
                                  non_linear_fn=None, bias=True)

            preds_i = tf.slice(preds_i, [0, 0, 0, 0], [-1, col_dim_i, -1, -1])
            preds_i = preds_i * emb_scale
            # preds_i = preds_i[:, :-(i + 1), :, :] * emb_scale
            preds_i = tf.reshape(preds_i, [-1, target_dim])

            logits = tf.matmul(preds_i, targets, transpose_b=True)

            print('logits: ' + str(logits.get_shape().as_list()))

            b = [x // (col_dim_i * row_dim) for x in range(total_elements)]
            #print(b)
            col = [x % (col_dim_i * row_dim) for x in range(total_elements)]
            #print(col)
            b = np.array(b)
            col = np.array(col)

            labels = b * col_dim * row_dim + (i + 1) * row_dim + col

            print('Labels: ', labels)

            onehot_labels = []

            for idx in labels:
                onehot = np.zeros(batch_dim * col_dim * row_dim)
                onehot[idx] = 1
                onehot_labels.append(onehot)

            onehot_labels = np.array(onehot_labels)
            onehot_labels = tf.constant(onehot_labels)
            #print('Onehot: ', onehot_labels[0])

            #loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=onehot_labels, logits=logits))
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=onehot_labels, logits=logits))

        return loss, logits


def add_residual_dense_block(in_layer, filter_dims, num_layers, act_func=tf.nn.relu, use_bn=False, bn_phaze=False,
                             scope='residual_dense_block', use_dilation=False, stochastic_depth=False,
                             stochastic_survive=0.9):
    with tf.variable_scope(scope):
        l = in_layer
        input_dims = in_layer.get_shape().as_list()
        num_channel_in = input_dims[-1]
        num_channel_out = filter_dims[-1]

        dilation = [1, 1, 1, 1]

        if use_dilation == True:
            dilation = [1, 2, 2, 1]

        layers.conv(l, scope='bt_conv', filter_dims=[1, 1, num_channel_out / 4], stride_dims=[1, 1], dilation=[1, 1, 1, 1],
                    non_linear_fn=None, bias=False, sn=False)

        for i in range(num_layers):
            l = layers.add_dense_layer(l, filter_dims=[filter_dims[0], filter_dims[1], num_channel_out / 4], use_bn=use_bn, act_func=act_func, bn_phaze=bn_phaze,
                                       scope='layer' + str(i), dilation=dilation)

        l = layers.add_dense_transition_layer(l, filter_dims=[1, 1, num_channel_out], act_func=act_func,
                                              scope='dense_transition_1', use_bn=use_bn, bn_phaze=bn_phaze, use_pool=False)

        pl = tf.constant(stochastic_survive)

        def train_mode():
            survive = tf.less(pl, tf.random_uniform(shape=[], minval=0.0, maxval=1.0))
            return tf.cond(survive, lambda: tf.add(l, in_layer), lambda: in_layer)

        def test_mode():
            return tf.add(tf.multiply(pl, l), in_layer)

    if stochastic_depth == True:
        return tf.cond(bn_phaze, train_mode, test_mode)

    return tf.add(l, in_layer)


def task(x, activation='relu', output_dim=256, scope='task_network', reuse=False, use_bn=True, bn_phaze=False, keep_prob=0.5):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        if activation == 'swish':
            act_func = util.swish
        elif activation == 'relu':
            act_func = tf.nn.relu
        elif activation == 'lrelu':
            act_func = tf.nn.leaky_relu
        else:
            act_func = tf.nn.sigmoid

        print('Task Layer1: ' + str(x.get_shape().as_list()))

        l = x

        #l = layers.self_attention(l, dense_block_depth)

        l = layers.conv(l, scope='conv1', filter_dims=[3, 3, dense_block_depth], stride_dims=[1, 1],
                        non_linear_fn=None, bias=False, dilation=[1, 1, 1, 1])

        l = add_residual_dense_block(l, filter_dims=[3, 3, dense_block_depth], num_layers=2,
                                    act_func=act_func, use_bn=use_bn, bn_phaze=bn_phaze, scope='block_1',
                                    stochastic_depth=False, stochastic_survive=0.7)

        l = add_residual_dense_block(l, filter_dims=[3, 3, dense_block_depth], num_layers=2,
                                    act_func=act_func, use_bn=use_bn, bn_phaze=bn_phaze, scope='block_2',
                                    stochastic_depth=False, stochastic_survive=0.65)

        l = add_residual_dense_block(l, filter_dims=[3, 3, dense_block_depth], num_layers=2,
                                    act_func=act_func, use_bn=use_bn, bn_phaze=bn_phaze, scope='block_3',
                                    stochastic_depth=False, stochastic_survive=0.6)

        l = add_residual_dense_block(l, filter_dims=[3, 3, dense_block_depth], num_layers=2,
                                     act_func=act_func, use_bn=use_bn, bn_phaze=bn_phaze, scope='block_4',
                                     stochastic_depth=False, stochastic_survive=0.6)

        if use_bn == True:
             l = layers.batch_norm_conv(l, b_train=bn_phaze, scope='bn1')

        l = act_func(l)

        # 1/2
        l = layers.conv(l, scope='conv2', filter_dims=[3, 3, dense_block_depth * 2], stride_dims=[2, 2],
                        non_linear_fn=None, bias=False, dilation=[1, 1, 1, 1])

        l = add_residual_dense_block(l, filter_dims=[3, 3, dense_block_depth * 2], num_layers=2,
                                    act_func=act_func, use_bn=use_bn, bn_phaze=bn_phaze, scope='block_5',
                                    stochastic_depth=False, stochastic_survive=0.6)

        l = add_residual_dense_block(l, filter_dims=[3, 3, dense_block_depth * 2], num_layers=2,
                                    act_func=act_func, use_bn=use_bn, bn_phaze=bn_phaze, scope='block_6',
                                    stochastic_depth=False, stochastic_survive=0.6)

        l = add_residual_dense_block(l, filter_dims=[3, 3, dense_block_depth * 2], num_layers=2,
                                    act_func=act_func, use_bn=use_bn, bn_phaze=bn_phaze, scope='block_7',
                                    stochastic_depth=False, stochastic_survive=0.6)

        l = add_residual_dense_block(l, filter_dims=[3, 3, dense_block_depth * 2], num_layers=2,
                                    act_func=act_func, use_bn=use_bn, bn_phaze=bn_phaze, scope='block_8',
                                    stochastic_depth=False, stochastic_survive=0.5)

        if use_bn == True:
            l = layers.batch_norm_conv(l, b_train=bn_phaze, scope='bn2')

        l = act_func(l)

        # 1/4
        l = layers.conv(l, scope='conv3', filter_dims=[3, 3, dense_block_depth * 4], stride_dims=[2, 2],
                        non_linear_fn=None, bias=False, dilation=[1, 1, 1, 1])

        l = add_residual_dense_block(l, filter_dims=[3, 3, dense_block_depth * 4], num_layers=2,
                                    act_func=act_func, use_bn=use_bn, bn_phaze=bn_phaze, scope='block_9',
                                    stochastic_depth=True, stochastic_survive=0.5)

        l = add_residual_dense_block(l, filter_dims=[3, 3, dense_block_depth * 4], num_layers=2,
                                    act_func=act_func, use_bn=use_bn, bn_phaze=bn_phaze, scope='block_10')

        l = act_func(l)

        latent = layers.global_avg_pool(l, output_length=output_dim)

    return latent


def encoder(x, activation='relu', scope='encoder_network', bn_phaze=False, use_bn=False):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):

        if activation == 'swish':
            act_func = util.swish
        elif activation == 'relu':
            act_func = tf.nn.relu
        elif activation == 'lrelu':
            act_func = tf.nn.leaky_relu
        else:
            act_func = tf.nn.sigmoid

        # [24 x 24]
        print('layer1: ' + str(x.get_shape().as_list()))
        l = layers.conv(x, scope='conv1', filter_dims=[3, 3, dense_block_depth], stride_dims=[1, 1],
                       non_linear_fn=None, bias=False, dilation=[1, 1, 1, 1])

        l = layers.self_attention(l, dense_block_depth)

        l = add_residual_dense_block(l, filter_dims=[3, 3, dense_block_depth], num_layers=2,
                                    act_func=act_func, use_bn=use_bn, bn_phaze=bn_phaze, scope='block_1',
                                    stochastic_depth=False, stochastic_survive=0.7)

        l = add_residual_dense_block(l, filter_dims=[3, 3, dense_block_depth], num_layers=2,
                                    act_func=act_func, use_bn=use_bn,  bn_phaze=bn_phaze, scope='block_2',
                                    stochastic_depth=False, stochastic_survive=0.65)

        l = add_residual_dense_block(l, filter_dims=[3, 3, dense_block_depth], num_layers=2,
                                    act_func=act_func, use_bn=use_bn, bn_phaze=bn_phaze, scope='block_3',
                                    stochastic_depth=False, stochastic_survive=0.6)

        l = add_residual_dense_block(l, filter_dims=[3, 3, dense_block_depth], num_layers=2,
                                     act_func=act_func, use_bn=use_bn, bn_phaze=bn_phaze, scope='block_4',
                                     stochastic_depth=False, stochastic_survive=0.6)

        if use_bn == True:
            l = layers.batch_norm_conv(l, b_train=bn_phaze, scope='bn1')

        l = act_func(l)

        # [12 x 12]
        l = layers.conv(l, scope='conv2', filter_dims=[3, 3, dense_block_depth * 2], stride_dims=[2, 2],
                       non_linear_fn=None, bias=False, dilation=[1, 1, 1, 1])

        print('layer2: ' + str(l.get_shape().as_list()))
        l = add_residual_dense_block(l, filter_dims=[3, 3, dense_block_depth * 2], num_layers=2,
                                    act_func=act_func, use_bn=use_bn, bn_phaze=bn_phaze, scope='block_5',
                                    stochastic_depth=False, stochastic_survive=0.6)

        l = add_residual_dense_block(l, filter_dims=[3, 3, dense_block_depth * 2], num_layers=2,
                                    act_func=act_func, use_bn=use_bn, bn_phaze=bn_phaze, scope='block_6',
                                    stochastic_depth=False, stochastic_survive=0.6)

        l = add_residual_dense_block(l, filter_dims=[3, 3, dense_block_depth * 2], num_layers=2,
                                    act_func=act_func, use_bn=use_bn, bn_phaze=bn_phaze, scope='block_7',
                                    stochastic_depth=False, stochastic_survive=0.6)

        l = add_residual_dense_block(l, filter_dims=[3, 3, dense_block_depth * 2], num_layers=2,
                                     act_func=act_func, use_bn=use_bn, bn_phaze=bn_phaze, scope='block_8',
                                     stochastic_depth=False, stochastic_survive=0.6)

        if use_bn == True:
            l = layers.batch_norm_conv(l, b_train=bn_phaze, scope='bn2')

        l = act_func(l)

        # [6 x 6]
        l = layers.conv(l, scope='conv5', filter_dims=[3, 3, dense_block_depth * 4], stride_dims=[2, 2],
                       non_linear_fn=None, bias=False, dilation=[1, 1, 1, 1])

        print('layer3: ' + str(l.get_shape().as_list()))
        l = add_residual_dense_block(l, filter_dims=[3, 3, dense_block_depth * 4], num_layers=2,
                                    act_func=act_func, use_bn=use_bn, bn_phaze=bn_phaze, scope='block_9',
                                    stochastic_depth=False, stochastic_survive=0.5)

        l = add_residual_dense_block(l, filter_dims=[3, 3, dense_block_depth * 4], num_layers=2,
                                    act_func=act_func, use_bn=use_bn, bn_phaze=bn_phaze, scope='block_10',
                                    stochastic_depth=True, stochastic_survive=0.5)

        last_dense_layer = l
        last_dense_layer = act_func(last_dense_layer)

    return last_dense_layer


def validate(model_path):
    label_list = os.listdir(label_directory)
    label_list.sort(key=str.lower)

    X = tf.placeholder(tf.float32, [batch_size * mini_batch_size, patch_height, patch_width, num_channel])
    Y = tf.placeholder(tf.float32, [batch_size, num_class_per_group])

    bn_train = tf.placeholder(tf.bool)

    # Launch the graph in a session
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    latent = encoder(X, activation='relu', use_bn=False, bn_phaze=bn_train, scope='encoder')
    print('Encoder latent: ' + str(latent.get_shape().as_list()))

    latent = layers.global_avg_pool(latent, output_length=representation_dim, use_bias=True, scope='gp')
    print('GP Dims: ' + str(latent.get_shape().as_list()))

    latent = tf.reshape(latent, [batch_size, num_context_patches, num_context_patches, -1])
    print('Latent Dims: ' + str(latent.get_shape().as_list()))

    latent = task(latent, output_dim=512, activation='relu', use_bn=True, bn_phaze=bn_train, scope='task')
    print('Task Latent Dims: ' + str(latent.get_shape().as_list()))

    prediction = layers.fc(latent, num_class_per_group, non_linear_fn=None, scope='fc_final')
    print('Prediction: ' + str(prediction.get_shape().as_list()))

    task_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='task') + tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope='fc_final')

    # softmax_temprature = 0.07
    softmax_temprature = 1.0

    class_loss = tf.reduce_mean(
        tf.losses.softmax_cross_entropy(onehot_labels=Y, logits=(prediction / softmax_temprature)))

    # training operation
    predict_op = tf.argmax(tf.nn.softmax(prediction), 1)
    confidence_op = tf.nn.softmax(prediction)

    # Launch the graph in a session
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        try:
            saver = tf.train.Saver()
            saver.restore(sess, model_path)
            print('Model Loaded')
        except:
            print('Model Load Failed')
            return

        print('Validation Data Directory: ' + test_data)

        for idx, labelname in enumerate(os.listdir(test_data)):
            if os.path.isdir(os.path.join(imgs_dirname, labelname).replace("\\", "/")) is False:
                continue

            test_label_dir = os.path.join(test_data, labelname).replace("\\", "/")
            img_files = os.listdir(test_label_dir)

            for f in img_files:
                bgrImg = cv2.imread(os.path.join(test_label_dir, f).replace("\\", "/"))
                img = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)

                img = img / 255.0
                patches = prepare_patches(img)

                pred, conf = sess.run([predict_op, confidence_op],
                              feed_dict={X: patches, bn_train: False})

                print(labelname + ', Prediction: ' + str(label_list[pred[0]]) + ', Confidence: ' + str(conf[0][pred[0]]))


def fine_tune(model_path):
    trX = []
    trY = []

    dir_list = os.listdir(imgs_dirname)
    dir_list.sort(key=str.lower)

    one_hot_length = len(os.listdir(imgs_dirname))

    with tf.device('/device:CPU:0'):
        X = tf.placeholder(tf.float32, [batch_size * mini_batch_size, patch_height, patch_width, num_channel])
        Y = tf.placeholder(tf.float32, [batch_size, num_class_per_group])

        for idx, labelname in enumerate(dir_list):
            imgs_list = load_images_from_folder(os.path.join(imgs_dirname, labelname), use_augmentation=True, add_noize=True)
            #print(trX.shape, imgs_list.shape)
            label = np.zeros(one_hot_length)

            if labelname == 'Unknown':
                label += (1.0 / num_class_per_group)
                print('label:', labelname, label)
            else:
                label[idx] += 1

            #print('label:', labelname, label)
            for idx2, img in enumerate(imgs_list):
                trY.append(label)
                trX.append(img)

        trX = np.array(trX)
        trX = trX.reshape((-1, input_height, input_width, num_channel))
        #print(trX.shape)

    bn_train = tf.placeholder(tf.bool)

    # Launch the graph in a session
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    latent = encoder(X, activation='relu', use_bn=False, bn_phaze=bn_train, scope='encoder')
    print('Encoder latent: ' + str(latent.get_shape().as_list()))

    latent = layers.global_avg_pool(latent, output_length=representation_dim, use_bias=True, scope='gp')
    print('GP Dims: ' + str(latent.get_shape().as_list()))

    latent = tf.reshape(latent, [batch_size, num_context_patches, num_context_patches, -1])
    print('Latent Dims: ' + str(latent.get_shape().as_list()))

    latent = task(latent, output_dim=512, activation='relu', use_bn=True, bn_phaze=bn_train, scope='task')
    print('Task Latent Dims: ' + str(latent.get_shape().as_list()))

    prediction = layers.fc(latent, num_class_per_group, non_linear_fn=None, scope='fc_final')
    print('Prediction: ' + str(prediction.get_shape().as_list()))

    task_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='task') + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='fc_final')

    #softmax_temprature = 0.07
    softmax_temprature = 1.0

    class_loss = tf.reduce_mean(
        tf.losses.softmax_cross_entropy(onehot_labels=Y, logits=(prediction / softmax_temprature)))
    # training operation
    predict_op = tf.argmax(tf.nn.softmax(prediction), 1)
    confidence_op = tf.nn.softmax(prediction)

    # Freeze Encoder & Transfer Mode
    #class_optimizer = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5).minimize(class_loss, var_list=task_vars)

    # Fine tune mode
    class_optimizer = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5).minimize(class_loss)

    # Launch the graph in a session
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        try:
            saver = tf.train.Saver()
            saver.restore(sess, model_path)
            print('Model Restored')
        except:
            try:
                variables_to_restore = [v for v in tf.trainable_variables()
                                        if v.name.split('/')[0] == 'encoder'
                                        or v.name.split('/')[0] == 'gp']
                saver = tf.train.Saver(variables_to_restore)
                saver.restore(sess, model_path)
                print('Partial Model Restored')
            except:
                print('Start New Training. Wait ...')

        for e in range(num_epoch):
            training_batches = zip(range(0, len(trX), batch_size),
                                   range(batch_size, len(trX) + 1, batch_size))

            trX, trY = shuffle(trX, trY)
            iteration = 0

            for start, end in training_batches:
                patches = np.empty([0, patch_height, patch_width, num_channel])

                for i in range(batch_size):
                    p = prepare_patches(trX[start + i])
                    patches = np.concatenate((patches, p), axis=0)

                _, l, c = sess.run([class_optimizer, class_loss, confidence_op],
                                   feed_dict={X: patches, Y: trY[start:end], bn_train: True})

                iteration = iteration + 1

                if iteration % 10 == 0:
                    print('epoch: ' + str(e) + ', loss: ' + str(l))

            try:
                saver = tf.train.Saver()
                saver.save(sess, model_path)
            except:
                print('Save failed')


def pretrain(model_path):
    trX = np.empty([0, input_height, input_width, 3], dtype=int)

    dir_list = os.listdir(imgs_dirname)
    dir_list.sort(key=str.lower)

    print('Please wait. It takes several minutes. Do not quit!')

    with tf.device('/device:CPU:0'):
        X = tf.placeholder(tf.float32, [batch_size * mini_batch_size, patch_height, patch_width, num_channel])

        for idx, labelname in enumerate(dir_list):
            imgs_list = load_images_from_folder(os.path.join(imgs_dirname, labelname), use_augmentation=True, add_noize=True)
            #print(trX.shape, imgs_list.shape)
            trX = np.concatenate((trX, imgs_list), axis=0)

        #print(trX.shape)
        trX = trX.reshape((-1, input_height, input_width, num_channel))

    bn_train = tf.placeholder(tf.bool)

    # Launch the graph in a session
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    context = encoder(X, activation='relu', use_bn=False, bn_phaze=bn_train, scope='encoder')
    print('Encoder Dims: ' + str(context.get_shape().as_list()))

    context = layers.global_avg_pool(context, output_length=representation_dim, use_bias=True, scope='gp')
    print('GP Dims: ' + str(context.get_shape().as_list()))

    context = tf.reshape(context, [batch_size, num_context_patches, num_context_patches, -1])
    print('Context Dims: ' + str(context.get_shape().as_list()))

    cpc_loss, cpc_logits = CPC(context, scope='cpc')

    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5).minimize(cpc_loss)

    softmax_cpc_logits = tf.nn.softmax(logits=cpc_logits)

    # Launch the graph in a session
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        try:
            saver = tf.train.Saver()
            saver.restore(sess, model_path)
            print('Model Restored')
        except:
            print('Start New Training. Wait ...')

        for e in range(num_epoch):
            training_batches = zip(range(0, len(trX), batch_size),
                                   range(batch_size, len(trX) + 1, batch_size))

            trX = shuffle(trX)
            iteration = 0

            for start, end in training_batches:
                patches = np.empty([0, patch_height, patch_width, num_channel])

                for i in range(batch_size):
                    p = prepare_patches(trX[start + i])
                    patches = np.concatenate((patches, p), axis=0)

                _, l, s_logit, c_logits = sess.run([optimizer, cpc_loss, softmax_cpc_logits, cpc_logits],
                                feed_dict={X: patches, bn_train: True})

                iteration = iteration + 1

                if iteration % 10 == 0:
                    #print('epoch: ' + str(e) + ', loss: ' + str(l) + ', softmax: ' + str(s_logit[0]) + ', logit: ' + str(c_logits[0]))
                    print('epoch: ' + str(e) + ', loss: ' + str(l))

            try:
                saver.save(sess, model_path)
            except:
                print('Save failed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, help='train/test/reps', default='train')
    parser.add_argument('--model_path', type=str, help='model check point file path', default='./model/m.ckpt')
    parser.add_argument('--data', type=str, help='data source base directory', default='./input')
    parser.add_argument('--out', type=str, help='output directory', default='./out/embedding')
    parser.add_argument('--train_data', type=str, help='training data directory', default='input')
    parser.add_argument('--test_data', type=str, help='test data directory', default='./test_data')
    parser.add_argument('--label', type=str, help='training data directory', default='input')
    parser.add_argument('--align', type=bool, help='use face alignment', default=False)

    args = parser.parse_args()

    mode = args.mode
    model_path = args.model_path

    imgs_dirname = args.train_data
    label_directory = args.label
    test_data = args.test_data

    input_height = 96
    input_width = 96
    num_channel = 3

    patch_height = 24
    patch_width = 24

    patch_size = 24

    dense_block_depth = 128
    representation_dim = 1024
    num_context_patches = 7

    scale_size = 110
    num_aug_patch = 4
    num_epoch = 10
    batch_size = 8
    mini_batch_size = num_context_patches * num_context_patches

    if mode == 'train':
        num_class_per_group = len(os.listdir(imgs_dirname))
        num_epoch = 50
        pretrain(model_path)
    elif mode == 'fine_tune':
        num_class_per_group = len(os.listdir(imgs_dirname))
        num_epoch = 20
        fine_tune(model_path)
    elif mode == 'validate':
        num_class_per_group = len(os.listdir(label_directory))
        batch_size = 1
        validate(model_path)
