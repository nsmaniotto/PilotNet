#!/usr/bin/python2
# -*- coding: utf-8 -*-

import os
import tensorflow.compat.v1 as tf
from tensorflow.core.protobuf import saver_pb2
from preprocess.imageSteeringDB import ImageSteeringDB
from nets.pilotNet import PilotNet

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    'dataset_dir', './data/datasets/driving_dataset',
    """Directory that stores input recored front view images and steering wheel angles.""")
tf.app.flags.DEFINE_bool(
    'clear_log', False,
    """force to clear old logs if exist.""")
"""
tf.app.flags.DEFINE_string(
    'log_dir', './logs/',
    Directory for training logs, including training summaries as well as training model checkpoint.)
"""
tf.app.flags.DEFINE_float(
    'L2NormConst', 1e-3,
    """L2-Norm const value for loss computation.""")
tf.app.flags.DEFINE_float(
    'learning_rate', 1e-4,
    """Learning rate determines the incremental steps of the values to find the best weight values.""")

"""
  1> Epoch：One Epoch is when an ENTIRE dataset is passed forward and backward through the neural network only ONCE,
    1个epoch等于使用训练集中的全部样本训练一次
  2> Batch Size：Total number of training examples present in a single batch,
    批大小, 每次训练在训练集中取Batch Size个样本训练
  3> Iteration: 1个iteration等于使用Batch Size个样本训练一次
"""
tf.app.flags.DEFINE_integer(
    'num_epochs', 30,
    """The numbers of epochs for training, train over the dataset about 30 times.""")
tf.app.flags.DEFINE_integer(
    'batch_size', 128,
    """The numbers of training examples present in a single batch for every training.""")

def train(argv=None):
    """Train PilotNet model"""

    # delete old logs
    if FLAGS.clear_log:
        if tf.gfile.Exists(FLAGS.log_dir):
            tf.gfile.DeleteRecursively(FLAGS.log_dir)
        tf.gfile.MakeDirs(FLAGS.log_dir)

    with tf.Graph().as_default():
        # construct model
        model = PilotNet()

        # images of the road ahead and steering angles in random order
        dataset = ImageSteeringDB(FLAGS.dataset_dir)

        '''
        采用随机梯度下降法进行训练(自适应估计方法, Adaptive Moment Estimation, Adam)：找到最好的权重值和偏差,以最小化输出误差
          1.计算梯度 compute_gradients(loss, <list of variables>)
          2.运用梯度 apply_gradients(<list of variables>)
        '''
        train_vars = tf.trainable_variables()
        # define loss
        loss = tf.reduce_mean(tf.square(tf.subtract(model.y_, model.steering))) \
               + tf.add_n([tf.nn.l2_loss(v) for v in train_vars]) * FLAGS.L2NormConst
        optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(loss)

        '''
        TensorFlow's V1 checkpoint format has been deprecated.
        Consider switching to the more efficient V2 format, now on by default.
        '''
        # saver = tf.train.Saver(write_version=saver_pb2.SaverDef.V1)
        saver = tf.train.Saver(tf.all_variables())
        # create a summary to monitor cost tensor
        tf.summary.scalar("loss", loss)
        # merge all summaries into a single op
        merged_summary_op = tf.summary.merge_all()
        init = tf.initialize_all_variables()

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            sess.run(init)

            # op to write logs to Tensorboard
            summary_writer = tf.summary.FileWriter(FLAGS.log_dir, graph=sess.graph)
            save_model_path = FLAGS.log_dir + "/checkpoint/"

            print("Run the command line:\n" \
                  "--> tensorboard --logdir={} " \
                  "\nThen open http://0.0.0.0:6006/ into your web browser".format(FLAGS.log_dir))

            # 在神经网络术语中: 一次epoch=一个向前传递(得到输出值)和一个向后传递(更新权重)
            for epoch in range(FLAGS.num_epochs):
                # 在整个神经网络运行, 你要传递两件事：损失计算和优化步骤

                # the number of batches is equal to number of iterations for one epoch.
                num_batches = int(dataset.num_images / FLAGS.batch_size)
                for batch in range(num_batches):
                    imgs, angles = dataset.load_train_batch(FLAGS.batch_size)

                    # Run optimization op (backprop) and cost op (to get loss value)
                    sess.run(
                        [loss, optimizer],
                        feed_dict={
                            model.image_input: imgs,
                            model.y_: angles,
                            model.keep_prob: 0.8
                        }
                    )

                    if batch % 10 == 0:
                        imgs, angles = dataset.load_val_batch(FLAGS.batch_size)
                        loss_value = sess.run(
                            loss,
                            feed_dict={
                                model.image_input: imgs,
                                model.y_: angles,
                                model.keep_prob: 1.0
                            }
                        )
                        # A training step is one gradient update, in one step batch_size many examples are processed.
                        print("Epoch: %d, Step: %d, Loss: %g" % (epoch, epoch * FLAGS.batch_size + batch, loss_value))

                    # write logs at every iteration
                    summary = merged_summary_op.eval(
                        feed_dict={
                            model.image_input: imgs,
                            model.y_: angles,
                            model.keep_prob: 1.0
                        }
                    )
                    # add loss summary for each batch
                    summary_writer.add_summary(summary, epoch * num_batches + batch)
                    # force tensorflow to synchronise summaries
                    summary_writer.flush()

                    # Save the model checkpoint periodically.
                    if batch % FLAGS.batch_size == 0:
                        if not os.path.exists(save_model_path):
                            os.makedirs(save_model_path)
                        checkpoint_path = os.path.join(save_model_path, "model.ckpt")
                        filename = saver.save(sess, checkpoint_path)

                print("Model saved in file: %s" % filename)

if __name__ == '__main__':
    # run the train function
    tf.app.run(main=train, argv=[])
