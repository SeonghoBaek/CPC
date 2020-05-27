# Tensorflow Implementation of CPC v2: Data Efficient Image Recognition with CPC
# Author: Seongho Baek 
# seonghobaek@gmail.com


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


def prepare_patches_from_file(file_name, patch_size=[24, 24], patch_dim=[7, 7], stride=12, use_augmentation=True):
    imgs = load_images(file_name, use_augmentation=use_augmentation)
    patches = np.empty([0, patch_height, patch_width, num_channel])

    for img in imgs:
        p = prepare_patches(img, patch_size=[patch_height, patch_width],
                            patch_dim=[num_context_patches, num_context_patches], stride=patch_height // 2)
        patches = np.concatenate((patches, p), axis=0)

    return patches


def load_images(file_name, use_augmentation=False, add_noise=False):
    images = []
    jpg_img = cv2.imread(file_name)
    img = cv2.cvtColor(jpg_img, cv2.COLOR_BGR2RGB)  # To RGB format
    img = cv2.resize(img, dsize=(input_height, input_width))

    if img is not None:
        img = np.array(img)

        n_img = img / 255.0
        images.append(n_img)

        if use_augmentation is True:
            n_img = cv2.flip(img, 1)
            n_img = n_img / 255.0
            images.append(n_img)

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

                if add_noise is True:
                    croped = croped + np.random.normal(size=(input_height, input_width, num_channel))

                    croped[croped > 255.0] = 255.0
                    croped[croped < 0] = 0.0

                croped = croped / 255.0
                images.append(croped)

    return np.array(images)


def load_images_from_folder(folder, use_augmentation=False, add_noise=False):
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

            if use_augmentation is True:
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

                    if add_noise is True:
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

        def context_transform(input_layer, scope='context_transform'):
            l = layers.conv(input_layer, scope=scope + '_1', filter_dims=[1, 1, target_dim], stride_dims=[1, 1],
                        non_linear_fn=tf.nn.relu, bias=True)

            l = layers.conv(l, scope=scope + '_2', filter_dims=[1, 1, target_dim], stride_dims=[1, 1],
                            non_linear_fn=None, bias=True)

            return l

        targets = context_transform(latents, 'transform')

        batch_dim, col_dim, row_dim = targets.get_shape().as_list()[:-1]
        #print(batch_dim, col_dim, row_dim)
        targets = tf.reshape(targets, [-1, target_dim])

        for i in range(steps_to_ignore, steps_to_predict):
            col_dim_i = col_dim - i - 1
            total_elements = batch_dim * col_dim_i * row_dim
            preds_i = context_transform(latents, 'transform')
            preds_i = tf.slice(preds_i, [0, 0, 0, 0], [-1, col_dim_i, -1, -1])
            preds_i = preds_i * emb_scale
            # preds_i = preds_i[:, :-(i + 1), :, :] * emb_scale
            preds_i = tf.reshape(preds_i, [-1, target_dim])

            logits = tf.matmul(preds_i, targets, transpose_b=True)

            print(str(i) + ', logits: ' + str(logits.get_shape().as_list()))

            b = [x // (col_dim_i * row_dim) for x in range(total_elements)]
            #print(b)
            col = [x % (col_dim_i * row_dim) for x in range(total_elements)]
            #print(col)
            b = np.array(b)
            col = np.array(col)

            labels = b * col_dim * row_dim + (i + 1) * row_dim + col

            print(str(i) + ', Labels: ', labels)

            onehot_labels = []

            for idx in labels:
                onehot = np.zeros(batch_dim * col_dim * row_dim)
                onehot[idx] = 1
                onehot_labels.append(onehot)

            onehot_labels = np.array(onehot_labels)
            onehot_labels = tf.constant(onehot_labels)
            #print('Onehot: ', onehot_labels[0])

            loss = loss + tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=onehot_labels, logits=logits))

        return loss, logits


def task(x, activation='relu', output_dim=256, scope='task_network', norm='layer', b_train=False):
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

        block_depth = dense_block_depth
        l = x
        l = layers.conv(l, scope='conv1', filter_dims=[3, 3, block_depth], stride_dims=[1, 1],
                        non_linear_fn=None, bias=False, dilation=[1, 1, 1, 1])

        if norm == 'layer':
            l = layers.layer_norm(l, scope='ln1')
        elif norm == 'batch':
            l = layers.batch_norm_conv(l, b_train=b_train, scope='bn1')

        l = act_func(l)

        for i in range(15):
            l = layers.add_residual_block(l,  filter_dims=[3, 3, block_depth], num_layers=2, act_func=act_func,
                                          norm=norm, b_train=b_train, scope='block1_' + str(i))

        latent = layers.global_avg_pool(l, output_length=output_dim)

    return latent


def encoder(x, activation='relu', scope='encoder_network', norm='layer', b_train=False):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):

        if activation == 'swish':
            act_func = util.swish
        elif activation == 'relu':
            act_func = tf.nn.relu
        elif activation == 'lrelu':
            act_func = tf.nn.leaky_relu
        else:
            act_func = tf.nn.sigmoid

        # [192 x 192]
        block_depth = dense_block_depth // 4

        l = layers.conv(x, scope='conv1', filter_dims=[5, 5, block_depth], stride_dims=[1, 1],
                       non_linear_fn=None, bias=False, dilation=[1, 1, 1, 1])

        if norm == 'layer':
            l = layers.layer_norm(l, scope='ln0')
        elif norm == 'batch':
            l = layers.batch_norm_conv(l, b_train=b_train, scope='bn0')

        l = act_func(l)

        for i in range(4):
            l = layers.add_residual_dense_block(l, filter_dims=[3, 3, block_depth], num_layers=2,
                                                act_func=act_func, norm=norm, b_train=b_train,
                                                scope='dense_block_1_' + str(i))

        # [64 x 64]
        block_depth = block_depth * 2

        l = layers.conv(l, scope='tr1', filter_dims=[3, 3, block_depth], stride_dims=[2, 2], non_linear_fn=None)

        if norm == 'layer':
            l = layers.layer_norm(l, scope='ln1')
        elif norm == 'batch':
            l = layers.batch_norm_conv(l, b_train=b_train, scope='bn1')

        l = act_func(l)

        print('Encoder Block 1: ' + str(l.get_shape().as_list()))

        for i in range(2):
            l = layers.add_residual_block(l, filter_dims=[3, 3, block_depth], num_layers=2, act_func=act_func,
                                          norm=norm, b_train=b_train, scope='res_block_1_' + str(i))

        # [32 x 32]
        block_depth = block_depth * 2

        l = layers.conv(l, scope='tr2', filter_dims=[3, 3, block_depth], stride_dims=[2, 2], non_linear_fn=None)

        if norm == 'layer':
            l = layers.layer_norm(l, scope='ln2')
        elif norm == 'batch':
            l = layers.batch_norm_conv(l, b_train=b_train, scope='bn2')

        l = act_func(l)

        print('Encoder Block 2: ' + str(l.get_shape().as_list()))

        for i in range(2):
            l = layers.add_residual_block(l, filter_dims=[3, 3, block_depth], num_layers=2, act_func=act_func,
                                          norm=norm, b_train=b_train, scope='res_block_2_' + str(i))

        # [16 x 16]
        block_depth = block_depth * 2

        l = layers.conv(l, scope='tr3', filter_dims=[3, 3, block_depth], stride_dims=[2, 2], non_linear_fn=None)

        if norm == 'layer':
            l = layers.layer_norm(l, scope='ln3')
        elif norm == 'batch':
            l = layers.batch_norm_conv(l, b_train=b_train, scope='bn3')

        l = act_func(l)

        print('Encoder Block 3: ' + str(l.get_shape().as_list()))

        for i in range(2):
            l = layers.add_residual_block(l, filter_dims=[3, 3, block_depth], num_layers=2, act_func=act_func,
                                          norm=norm, b_train=b_train, scope='res_block_3' + str(i))

        # [8 x 8]
        block_depth = block_depth * 2
        l = layers.conv(l, scope='tr4', filter_dims=[3, 3, block_depth], stride_dims=[2, 2], non_linear_fn=None)

        if norm == 'layer':
            l = layers.layer_norm(l, scope='ln4')
        elif norm == 'batch':
            l = layers.batch_norm_conv(l, b_train=b_train, scope='bn4')

        l = act_func(l)

        print('Encoder Block 4: ' + str(l.get_shape().as_list()))

        for i in range(2):
            l = layers.add_residual_block(l, filter_dims=[3, 3, block_depth], num_layers=2, act_func=act_func,
                                          norm=norm, b_train=b_train, use_dilation=True, scope='res_block_4_' + str(i))

        # [4 x 4]
        block_depth = block_depth * 2
        l = layers.conv(l, scope='tr5', filter_dims=[3, 3, block_depth], stride_dims=[2, 2], non_linear_fn=None)
        print('Encoder Block 5: ' + str(l.get_shape().as_list()))

        if norm == 'layer':
            l = layers.layer_norm(l, scope='ln5')
        elif norm == 'batch':
            l = layers.batch_norm_conv(l, b_train=b_train, scope='bn5')

        l = act_func(l)

        for i in range(2):
            l = layers.add_residual_block(l, filter_dims=[3, 3, block_depth], num_layers=2, act_func=act_func,
                                          norm=norm, b_train=b_train, use_dilation=True, scope='res_block_5_' + str(i))

        last_layer = l

        context = layers.global_avg_pool(last_layer, output_length=representation_dim, use_bias=True, scope='gp')
        print('Encoder GP Dims: ' + str(context.get_shape().as_list()))

        context = tf.reshape(context, [batch_size, num_context_patches, num_context_patches, -1])
        print('Context Dims: ' + str(context.get_shape().as_list()))

    return context


def validate(model_path):
    label_list = os.listdir(label_directory)
    label_list.sort(key=str.lower)

    X = tf.placeholder(tf.float32, [batch_size * mini_batch_size, patch_height, patch_width, num_channel])
    Y = tf.placeholder(tf.float32, [batch_size, num_class_per_group])

    b_train = tf.placeholder(tf.bool)

    # Launch the graph in a session
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    latent = encoder(X, activation='relu', norm='layer', b_train=b_train, scope='encoder')
    print('Encoder latent: ' + str(latent.get_shape().as_list()))

    latent = task(latent, output_dim=512, activation='relu', norm='batch', b_train=b_train, scope='task')
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
                patches = prepare_patches(img, patch_size=[patch_height, patch_width],
                                    patch_dim=[num_context_patches, num_context_patches], stride=patch_height // 2)
                pred, conf = sess.run([predict_op, confidence_op],
                              feed_dict={X: patches, b_train: False})

                print(labelname + ', Prediction: ' + str(label_list[pred[0]]) + ', Confidence: ' + str(conf[0][pred[0]]))


def fine_tune(model_path, b_freeze=False):
    trX = []
    trY = []

    dir_list = os.listdir(imgs_dirname)
    dir_list.sort(key=str.lower)

    one_hot_length = len(os.listdir(imgs_dirname))

    with tf.device('/device:CPU:0'):
        X = tf.placeholder(tf.float32, [batch_size * mini_batch_size, patch_height, patch_width, num_channel])
        Y = tf.placeholder(tf.float32, [batch_size, num_class_per_group])

        for idx, labelname in enumerate(dir_list):
            directory = os.path.join(imgs_dirname, labelname).replace("\\", "/")
            imgs_file_list = os.listdir(directory)
            label = np.zeros(one_hot_length)
            label[idx] += 1
            print(labelname + ': '+ str(idx))
            for file in imgs_file_list:
                trY.append(label)
                trX.append(os.path.join(directory, file).replace("\\", "/"))

    b_train = tf.placeholder(tf.bool)

    # Launch the graph in a session
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    latent = encoder(X, activation='relu', norm='layer', b_train=b_train, scope='encoder')
    print('Encoder latent: ' + str(latent.get_shape().as_list()))

    latent = task(latent, output_dim=512, activation='relu', norm='batch', b_train=b_train, scope='task')
    print('Task Latent Dims: ' + str(latent.get_shape().as_list()))

    prediction = layers.fc(latent, num_class_per_group, non_linear_fn=None, scope='fc_final')
    print('Prediction: ' + str(prediction.get_shape().as_list()))

    task_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='task') + \
                tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='fc_final')

    #softmax_temprature = 0.07
    softmax_temprature = 1.0

    class_loss = tf.reduce_mean(
        tf.losses.softmax_cross_entropy(onehot_labels=Y, logits=(prediction / softmax_temprature)))
    # training operation
    predict_op = tf.argmax(tf.nn.softmax(prediction), 1)
    confidence_op = tf.nn.softmax(prediction)

    if b_freeze is True:
        # Freeze Encoder Weights & Transfer Mode
        class_optimizer = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5).minimize(class_loss, var_list=task_vars)
    else:
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
                    #p = prepare_patches(trX[start + i], patch_size=[patch_height, patch_width],
                    #                    patch_dim=[num_context_patches, num_context_patches], stride=patch_height // 2)
                    p = prepare_patches_from_file(trX[start + i], patch_size=[patch_height, patch_width],
                                        patch_dim=[num_context_patches, num_context_patches], stride=patch_height // 2, use_augmentation=False)
                    patches = np.concatenate((patches, p), axis=0)

                _, l, c = sess.run([class_optimizer, class_loss, confidence_op],
                                   feed_dict={X: patches, Y: trY[start:end], b_train: True})

                iteration = iteration + 1

                if iteration % 10 == 0:
                    print('epoch: ' + str(e) + ', loss: ' + str(l))

            try:
                saver = tf.train.Saver()
                saver.save(sess, model_path)
            except:
                print('Save failed')


def pretrain(model_path):
    dir_list = os.listdir(imgs_dirname)
    dir_list.sort(key=str.lower)

    print('Please wait. It takes several minutes. Do not quit!')

    with tf.device('/device:CPU:0'):
        X = tf.placeholder(tf.float32, [batch_size * mini_batch_size, patch_height, patch_width, num_channel])
        trX = dir_list

    b_train = tf.placeholder(tf.bool)

    # Launch the graph in a session
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    context = encoder(X, activation='relu', norm='layer', b_train=b_train, scope='encoder')
    print('Encoder Dims: ' + str(context.get_shape().as_list()))

    cpc_loss, cpc_logits = CPC(context, emb_scale=0.1, steps_to_ignore=0, steps_to_predict=2, scope='cpc')

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
            trX = shuffle(trX)
            iteration = 0

            for input_file in trX:
                fullname = os.path.join(imgs_dirname, input_file).replace("\\", "/")

                patches = prepare_patches_from_file(fullname, patch_size=[patch_height, patch_width], patch_dim=[num_context_patches, num_context_patches], stride=patch_height//2)
                _, l, s_logit, c_logits = sess.run([optimizer, cpc_loss, softmax_cpc_logits, cpc_logits],
                                                   feed_dict={X: patches, b_train: True})

                iteration = iteration + 1

                if iteration % 10 == 0:
                    # print('epoch: ' + str(e) + ', loss: ' + str(l) + ', softmax: ' + str(s_logit[0]) + ', logit: ' + str(c_logits[0]))
                    print('epoch: ' + str(e) + ', loss: ' + str(l))

            try:
                saver.save(sess, model_path)
            except:
                print('Save failed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, help='train/fine_tune/validate', default='train')
    parser.add_argument('--model_path', type=str, help='Model check point file path', default='./model/m.ckpt')
    parser.add_argument('--train_data', type=str, help='Training data directory', default='input')
    parser.add_argument('--test_data', type=str, help='Test data directory', default='./test_data')
    parser.add_argument('--label', type=str, help='Label list directory. Used for identifying validate summary', default='input')

    args = parser.parse_args()

    mode = args.mode
    model_path = args.model_path

    imgs_dirname = args.train_data
    label_directory = args.label
    test_data = args.test_data

    # Input Data Dimension
    input_height = 384 # 96
    input_width = 384 # 96
    num_channel = 3

    # If you divide a image([height, width]) to [p_height, p_width] sized patch,
    # total counts of generated patches are (2 * (height//p_height) - 1)**2 (50% overwrap)
    # The patch width/height pixel counts should be even number.
    # Example) Input image size = [1024 x 1024]. A Patch size is [128 x 128], then,
    # 1024 // 128 = 8. Counts of total patches = (8 + 7)**2 = 225

    # Patch Dimension
    patch_height = 192 # 32
    patch_width = 192 # 32

    # Dense Conv Block Base Channel Depth
    dense_block_depth = 128

    # Bottle neck(depth narrow down) depth. See Residual Dense Block and Residual Block.
    bottleneck_depth = 64

    # CPC Encoding latent dimension
    representation_dim = 1024

    # Number of patches in horizontal / vertical
    # Total counts of patches are num_context_patches**2
    num_context_patches = 3

    # Input data augmentation Setting. See function: load_images_from_folder.
    scale_size = 396 # 110
    num_aug_patch = 2

    # Training parameter
    num_epoch = 10
    batch_size = 4

    # Mini batch
    # Each input in batch_size batches is devided to num_context_patches * num_context_patches patches.
    # ex) batch_size = 8, num_context_patches = 7
    #     Each input in batch are devided 7 * 7 patches.
    #     Total input data in 1 batch is (8 * 7 * 7) items.
    mini_batch_size = num_context_patches * num_context_patches

    if mode == 'train':
        # Train unsupervised CPC encoder.
        num_class_per_group = len(os.listdir(imgs_dirname))
        num_epoch = 20
        pretrain(model_path)
    elif mode == 'fine_tune':
        # Fine tune with specific downstream task.
        num_class_per_group = len(os.listdir(imgs_dirname))
        num_epoch = 20
        batch_size = 8
        fine_tune(model_path, b_freeze=True)
    elif mode == 'validate':
        num_class_per_group = len(os.listdir(label_directory))
        batch_size = 1
        validate(model_path)
