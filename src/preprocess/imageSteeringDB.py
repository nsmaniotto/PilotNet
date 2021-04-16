import scipy.misc
import random
from PIL import Image
import numpy as np

class ImageSteeringDB(object):
    """Preprocess images of the road ahead ans steering angles."""

    def __init__(self, data_dir):
        imgs = []
        angles = []

        # points to the end of the last batch, train & validation
        self.train_batch_pointer = 0
        self.val_batch_pointer = 0

        # read data.txt
        #data_path = data_dir + "/"
        data_path ="/scratch/rmoine/PIR/PilotNet/data/datasets/driving_dataset/"

        with open(data_path + "data.txt") as f:
            for line in f:
                imgs.append(data_path + line.split()[0])
                # the paper by Nvidia uses the inverse of the turning radius,
                # but steering wheel angle is proportional to the inverse of turning radius
                # so the steering wheel angle in radians is used as the output
                angles.append(float(line.split()[1]) * scipy.pi / 180)

        # shuffle list of images
        c = list(zip(imgs, angles))
        random.shuffle(c)
        imgs, angles = zip(*c)

        # get number of images
        self.num_images = len(imgs)

        self.train_imgs = imgs[:int(self.num_images * 0.8)]
        self.train_angles = angles[:int(self.num_images * 0.8)]

        self.val_imgs = imgs[-int(self.num_images * 0.2):]
        self.val_angles = angles[-int(self.num_images * 0.2):]

        self.num_train_images = len(self.train_imgs)
        self.num_val_images = len(self.val_imgs)

    def load_train_batch(self, batch_size):
        batch_imgs = []
        batch_angles = []
        for i in range(0, batch_size):
            # idea
            """
            Replace deprecated scipy.misc.imresize
            using pillow:
            numpy.array(Image.fromarray(arr).resize())
            """

            # old version
            """
            batch_imgs.append(scipy.misc.imresize(
                numpy.array(Image.fromarray(arr).resize())
            Image.fromarray(orj_img).resize(size=(new_h, new_w))
                scipy.misc.imread(self.train_imgs[(self.train_batch_pointer + i) % self.num_train_images])[-150:],
                [66, 200]) / 255.0)
            """

            # new version
            img_path = self.train_imgs[(self.train_batch_pointer + i) % self.num_train_images]
            img_path = img_path[-150:]
            img = Image.open(img_path).resize(size=(200, 66))
            img = np.array(img)
            batch_imgs.append(img / 255.0)
            batch_angles.append([self.train_angles[(self.train_batch_pointer + i) % self.num_train_images]])
        self.train_batch_pointer += batch_size
        return batch_imgs, batch_angles

    def load_val_batch(self, batch_size):
        batch_imgs = []
        batch_angles = []
        for i in range(0, batch_size):

            # idea
            """
            Replace deprecated scipy.misc.imresize
            using pillow:
            numpy.array(Image.fromarray(arr).resize())
            """


            # old version
            """
            batch_imgs.append(scipy.misc.imresize(scipy.misc.imread(self.val_imgs[(self.val_batch_pointer + i) % self.num_val_images])[-150:], [66, 200]) / 255.0)
            """

            img_path = self.val_imgs[(self.val_batch_pointer + i) % self.num_val_images]
            img_path = img_path[-150:]
            img = Image.open(img_path).resize(size=(200, 66))
            img = np.array(img)
            batch_imgs.append(img / 255.0)
            batch_angles.append([self.val_angles[(self.val_batch_pointer + i) % self.num_val_images]])
        self.val_batch_pointer += batch_size
        return batch_imgs, batch_angles
