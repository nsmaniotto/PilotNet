import scipy.misc
import random
from PIL import Image
import numpy as np
import tensorflow as tf

# from nuscenes_devkit.python_sdk.nuscenes import nuscenes
import json


class NuSceneDB(object):
    # Every label present in the nuScene dataset
    labels = ['animal',
            'human.pedestrian.adult',
            'human.pedestrian.child',
            'human.pedestrian.construction_worker',
            'human.pedestrian.personal_mobility',
            'human.pedestrian.police_officer',
            'human.pedestrian.stroller',
            'human.pedestrian.wheelchair',
            'movable_object.barrier',
            'movable_object.debris',
            'movable_object.pushable_pullable',
            'movable_object.trafficcone',
            'static_object.bicycle_rack',
            'vehicle.bicycle',
            'vehicle.bus.bendy',
            'vehicle.bus.rigid',
            'vehicle.car',
            'vehicle.construction',
            'vehicle.emergency.ambulance',
            'vehicle.emergency.police',
            'vehicle.motorcycle',
            'vehicle.trailer',
            'vehicle.truck']

    """Preprocess images of the road ahead and expected outputs."""

    def __init__(self, data_dir):
        imgs = []  # Will store the path of each image
        outputs = []  # Will store the occurrence of each label for each image

        # points to the end of the last batch, train & validation
        self.train_batch_pointer = 0
        self.val_batch_pointer = 0

        # Extract images annotations
        # data_path = data_dir + "/"
        data_path = "/scratch/rmoine/PIR/nuscene"
        json_path = "/scratch/rmoine/PIR/extracted_data_nusceneImage.json"

        with open(json_path) as json_file:
            images = json.loads(json_file.read())

            for image in images:
                filename = image['imageName']
                categories = image['categories']

                imagePath = data_path + filename

                if tf.gfile.Exists(imagePath):
                    imageOutputs = [0] * len(NuSceneDB.labels)

                    for detectedCategory in list(categories):
                        # Retrieve category array of bounding boxes
                        boundingBoxes = categories[detectedCategory]

                        # Count the number of occurrences for the detected category
                        occurrences = len(boundingBoxes)

                        # Retrieve index of the label within 'labels' array
                        labelIndex = NuSceneDB.labels.index(detectedCategory)

                        # Store the number of occurrences within the right index of the 'imageOutputs' array
                        imageOutputs[labelIndex] = occurrences

                    imgs.append(data_path + filename)
                    outputs.append(imageOutputs)

        # shuffle list of images
        c = list(zip(imgs, outputs))
        random.shuffle(c)
        imgs, outputs = zip(*c)

        # get number of images
        self.num_images = len(imgs)

        self.train_imgs = imgs[:int(self.num_images * 0.8)]
        self.train_outputs = outputs[:int(self.num_images * 0.8)]

        self.val_imgs = imgs[-int(self.num_images * 0.2):]
        self.val_outputs = outputs[-int(self.num_images * 0.2):]

        self.num_train_images = len(self.train_imgs)
        self.num_val_images = len(self.val_imgs)

    def load_train_batch(self, batch_size):
        batch_imgs = []
        batch_outputs = []

        # Resize every image to 66*200
        # Store their respective outputs
        # Return the resized images and their outputs
        for i in range(0, batch_size):
            img_path = self.train_imgs[(self.train_batch_pointer + i) % self.num_train_images]
            img_path = img_path[-150:]
            img = Image.open(img_path).resize(size=(200, 66))
            img = np.array(img)
            batch_imgs.append(img / 255.0)
            batch_outputs.append([self.train_outputs[(self.train_batch_pointer + i) % self.num_train_images]])

        self.train_batch_pointer += batch_size
        return batch_imgs, batch_outputs

    def load_val_batch(self, batch_size):
        batch_imgs = []
        batch_outputs = []

        # Resize every image to 66*200
        # Store their respective outputs
        # Return the resized images and their outputs
        for i in range(0, batch_size):
            img_path = self.val_imgs[(self.val_batch_pointer + i) % self.num_val_images]
            img_path = img_path[-150:]
            img = Image.open(img_path).resize(size=(200, 66))
            img = np.array(img)
            batch_imgs.append(img / 255.0)
            batch_outputs.append([self.val_outputs[(self.val_batch_pointer + i) % self.num_val_images]])
        self.val_batch_pointer += batch_size
        return batch_imgs, batch_outputs
