import tensorflow as tf
import scipy.misc
from nets.model import PilotNet
import cv2
from subprocess import call

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    'model', './data/model_nvidia/model.ckpt',
    """Path to the model parameter file.""")
tf.app.flags.DEFINE_string(
    'steer_image', './data/logo/steering_wheel_image.jpg',
    """Steering wheel image to show corresponding steering wheel angle.""")
tf.app.flags.DEFINE_string(
    'dataset_path', './data/dataset_nvidia/driving_dataset/',
    """Input front view image.""")

WIN_MARGIN_LEFT = 240
WIN_MARGIN_TOP = 240
WIN_MARGIN_BETWEEN = 180
WIN_WIDTH = 480

if __name__ == '__main__':


    img = cv2.imread(FLAGS.steer_image, 0)
    rows,cols = img.shape

    # Visualization init
    cv2.namedWindow("steering wheel", cv2.WINDOW_NORMAL)
    cv2.moveWindow("steering wheel", WIN_MARGIN_LEFT, WIN_MARGIN_TOP)
    cv2.namedWindow("scenario", cv2.WINDOW_NORMAL)
    cv2.moveWindow("scenario", WIN_MARGIN_LEFT+cols+WIN_MARGIN_BETWEEN, WIN_MARGIN_TOP)

    with tf.Graph().as_default():
        smoothed_angle = 0
        i = 0

        model = PilotNet()

        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, FLAGS.model)

            while(cv2.waitKey(10) != ord('q')):
                full_image = scipy.misc.imread(FLAGS.dataset_path + str(i) + ".jpg", mode="RGB")
                image = scipy.misc.imresize(full_image[-150:], [66, 200]) / 255.0

                steering = sess.run(
                    model.steering,
                    feed_dict={
                        model.image_input: [image],
                        model.keep_prob: 1.0
                    }
                )

                degrees = steering[0][0] * 180.0 / scipy.pi
                call("clear")
                print("Predicted steering angle: " + str(degrees) + " degrees")

                cv2.imshow("scenario", cv2.cvtColor(full_image, cv2.COLOR_RGB2BGR))
                print("Scenario image size: {} x {}").format(full_image.shape[0], full_image.shape[1])

                # make smooth angle transitions by turning the steering wheel based on the difference of the current angle
                # and the predicted angle
                smoothed_angle += 0.2 * pow(abs((degrees - smoothed_angle)), 2.0 / 3.0) * (degrees - smoothed_angle) / abs(degrees - smoothed_angle)
                M = cv2.getRotationMatrix2D((cols/2,rows/2), -smoothed_angle, 1)
                dst = cv2.warpAffine(img,M,(cols,rows))
                cv2.imshow("steering wheel", dst)

                i += 1

            cv2.destroyAllWindows()
