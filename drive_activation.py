import tensorflow as tf
import scipy.misc
import model_nvidia
import cv2
import matplotlib.pyplot as plt
import numpy as np



sess = tf.InteractiveSession()
saver = tf.train.Saver()
saver.restore(sess, "save_model_nvidia/model.ckpt")
print("Load session successfully")

#randomly choose an img from dataset
full_image = scipy.misc.imread("dataset_nvidia/" + "29649" + ".jpg", mode="RGB")
image = scipy.misc.imresize(full_image, [66, 200]) / 255.0

plt.imshow(image)
plt.show()

conv5act = model_nvidia.h_conv5.eval(feed_dict={model_nvidia.x: [image]})
conv4act = model_nvidia.h_conv4.eval(feed_dict={model_nvidia.x: [image]})
conv3act = model_nvidia.h_conv3.eval(feed_dict={model_nvidia.x: [image]})
conv2act = model_nvidia.h_conv2.eval(feed_dict={model_nvidia.x: [image]})
conv1act = model_nvidia.h_conv1.eval(feed_dict={model_nvidia.x: [image]})


# Get the mean, and supress the first(batch) dimension
averageC5 = np.mean(conv5act,axis=3).squeeze(axis=0)
averageC4 = np.mean(conv4act,axis=3).squeeze(axis=0)
averageC3 = np.mean(conv3act,axis=3).squeeze(axis=0)
averageC2 = np.mean(conv2act,axis=3).squeeze(axis=0)
averageC1 = np.mean(conv1act,axis=3).squeeze(axis=0)


#upscale
averageC5up = scipy.misc.imresize(averageC5,[averageC4.shape[0], averageC4.shape[1]])
multC45 = np.multiply(averageC5up,averageC4)
multC45up = scipy.misc.imresize(multC45,[averageC3.shape[0], averageC3.shape[1]])
multC34 = np.multiply(multC45up,averageC3)
multC34up = scipy.misc.imresize(multC34,[averageC2.shape[0], averageC2.shape[1]])
multC23 = np.multiply(multC34up,averageC2)
multC23up = scipy.misc.imresize(multC23,[averageC1.shape[0], averageC1.shape[1]])
multC12 = np.multiply(multC23up,averageC1)
multC12up = scipy.misc.imresize(multC12,[image.shape[0], image.shape[1]])

# normalize to [0,1], however, it did not show the salient map, the multC12up shows something like salient
salient_mask = (multC12up - np.min(multC12up))/(np.max(multC12up) - np.min(multC12up))
#plt.imshow(salient_mask)
plt.imshow(multC12up)
plt.show()





