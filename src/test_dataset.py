import tensorflow.compat.v1 as tf
import scipy.misc
from nets.pilotNet import PilotNet
from subprocess import call

from PIL import Image
import numpy as np

FLAGS = tf.app.flags.FLAGS

"""model from implemented training"""
tf.app.flags.DEFINE_string(
    'model_file', '../data/datasets/driving_dataset/checkpoint/model.ckpt',
    """Path to the model parameter file.""")

tf.app.flags.DEFINE_string(
    'dataset_dir', '../data/datasets/driving_dataset/sample',
    """Directory that stores input recorded front view images.""")

if __name__ == '__main__':

    # Open file which contains : input and expected output
    dataFile = open(FLAGS.dataset_dir + "/data.txt", "r")

    errorSum = 0.
    lineCount = 0

    with tf.Graph().as_default():

        # Construct model
        model = PilotNet()

        with tf.Session() as sess:
            # Restore model weights
            saver = tf.train.import_meta_graph(FLAGS.model_file + '.meta')
            saver.restore(sess, FLAGS.model_file)

            call("clear")

            for line in dataFile:
                lineCount += 1

                imageName, expectedOutput = line.split()

                full_image = Image.open(FLAGS.dataset_dir + "/" + imageName)
                img = full_image.resize(size=(200, 66))
                img = np.array(img)
                image = (img / 255.0)

                steering = sess.run(
                    model.steering,
                    feed_dict={
                        model.image_input: [image],
                        model.keep_prob: 1.0
                    }
                )

                degrees = steering[0][0] * 180.0 / scipy.pi

                errorSum += np.absolute(float(expectedOutput) - degrees)

                # print("[" + imageName + "] Error: " + str(np.absolute(float(expectedOutput) - degrees)) + "(Predicted " + str(degrees) + " | Expected: " + expectedOutput + ")")
                # print("Scenario image size: {} x {}").format(full_image.shape[0], full_image.shape[1])

    print("Average error based on the " + str(lineCount) + " images: " + str(errorSum / lineCount) + " degrees")
