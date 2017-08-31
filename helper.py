import re
import random
import numpy as np
import os.path
import scipy.misc
import shutil
import zipfile
import time
import tensorflow as tf
import time
import csv
import cv2

from glob import glob
from urllib.request import urlretrieve
from tqdm import tqdm

class DLProgress(tqdm):
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num

def maybe_download_pretrained_vgg(data_dir):
    """
    Download and extract pretrained vgg model if it doesn't exist
    :param data_dir: Directory to download the model to
    """
    vgg_filename = 'vgg.zip'
    vgg_path = os.path.join(data_dir, 'vgg')
    vgg_files = [
        os.path.join(vgg_path, 'variables/variables.data-00000-of-00001'),
        os.path.join(vgg_path, 'variables/variables.index'),
        os.path.join(vgg_path, 'saved_model.pb')]

    missing_vgg_files = [vgg_file for vgg_file in vgg_files if not os.path.exists(vgg_file)]
    if missing_vgg_files:
        # Clean vgg dir
        if os.path.exists(vgg_path):
            shutil.rmtree(vgg_path)
        os.makedirs(vgg_path)

        # Download vgg
        print('Downloading pre-trained vgg model...')
        with DLProgress(unit='B', unit_scale=True, miniters=1) as pbar:
            urlretrieve(
                'https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip',
                os.path.join(vgg_path, vgg_filename),
                pbar.hook)

        # Extract vgg
        print('Extracting model...')
        zip_ref = zipfile.ZipFile(os.path.join(vgg_path, vgg_filename), 'r')
        zip_ref.extractall(data_dir)
        zip_ref.close()

        # Remove zip file to save space
        os.remove(os.path.join(vgg_path, vgg_filename))

 def load_data(self, data_dir, validation_fraction):
    """
    Load the data and and return the sets which have paths to validation and training
    :param data_dir: Path to folder that contains all the datasets
    :param validation_fraction: Fraction to store for validation
    """
    images = data_dir + '/training/image_2/*.png'
    labels = data_dir + '/training/gt_image_2/*_road_*.png'

    # Get all the image and label paths
    image_paths = glob(images)
    label_paths = {
        os.path.basename(path).replace('_road_', '_'): path
        for path in glob(labels)}
    self.label_paths = label_paths

    # Check if there is an image folder
    num_images = len(image_paths)
    if num_images == 0:
        raise RuntimeError('No data files found in ' + data_dir)

    # Randomly shuffle before setting up the validation and training sets
    random.shuffle(image_paths)
    validation_images = image_paths[:int(validation_fraction*num_images)]
    training_images = image_paths[int(validation_fraction*num_images):]

    # Return the array of paths to images for validation and training sets
    return validation_images, training_images

def gen_batch_function(image_paths, image_shape):
    """
    Generate function to create batches of training data
    :param data_folder: Path to folder that contains all the datasets
    :param image_shape: Tuple - Shape of image
    :return:
    """
    def get_batches_fn(batch_size):
        """
        Create batches of training data
        :param batch_size: Batch Size
        :return: Batches of training data
        """
        # Randomly shuffle the images in the folder
        random.shuffle(image_paths)

        for batch_i in range(0, len(image_paths), batch_size):
            images = []
            gt_images = []
            for image_file in image_paths[batch_i:batch_i+batch_size]:
                gt_image_file = label_paths[os.path.basename(image_file)]

                # Reshape the images(original and ground truth)
                image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)
                gt_image = scipy.misc.imresize(scipy.misc.imread(gt_image_file), image_shape)

                # Add random rotation
                angle_to_rotate = np.random.uniform(-30, 30)
                image = scipy.misc.imrotate(image, angle_to_rotate)
                gt_image = scipy.misc.imrotate(gt_image, angle_to_rotate)

                # Add random translation
                # NOTE: image_shape[1] = width and image_shape[0] = height
                x_translate = np.random.uniform(-40, 40)
                y_translate = np.random.uniform(-40, 40)
                translate_matrix = np.float32([[1, 0, x_translate],[0, 1, y_translate]])
                image = cv2.warpAffine(image, translate_matrix, (image_shape[1], image_shape[0]))
		        # NOTE: Both the images(original and ground truth) have the same shape
                gt_image = cv2.warpAffine(gt_image, translate_matrix, (image_shape[1], image_shape[0]))

                # Add histogram equalization along the 'Y' channel of a YUV image
                # NOTE: This is supposed to help remove shadows in the image
                image_yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
                image_yuv[:,:,0] = cv2.equalizeHist(image_yuv[:,:,0])
                image = cv2.cvtColor(image_yuv, cv2.COLOR_YUV2RGB)

                # Array representing the ground truth in boolean form
                gt_bg = np.all(gt_image == background_color, axis=2)
                gt_bg = gt_bg.reshape(*gt_bg.shape, 1)
                gt_image = np.concatenate((gt_bg, np.invert(gt_bg)), axis=2)

                # Append the original images and the ground truth labels
                images.append(image)
                gt_images.append(gt_image)

            yield np.array(images), np.array(gt_images)
    return get_batches_fn

def gen_test_output(sess, logits, keep_prob, image_pl, data_folder, image_shape):
    """
    Generate test output using the test images
    :param sess: TF session
    :param logits: TF Tensor for the logits
    :param keep_prob: TF Placeholder for the dropout keep robability
    :param image_pl: TF Placeholder for the image placeholder
    :param data_folder: Path to the folder that contains the datasets
    :param image_shape: Tuple - Shape of image
    :return: Output for for each test image
    """
    for image_file in glob(os.path.join(data_folder, 'image_2', '*.png')):
        # Resize the image to the input image size
        image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)

        # Perform segmentation to the test image(add a mask reresenting predictions)
        im_softmax = sess.run([tf.nn.softmax(logits)],{keep_prob: 1.0, image_pl: [image]})
        im_softmax = im_softmax[0][:, 1].reshape(image_shape[0], image_shape[1])
        segmentation = (im_softmax > 0.5).reshape(image_shape[0], image_shape[1], 1)
        mask = np.dot(segmentation, np.array([[0, 255, 0, 127]]))
        mask = scipy.misc.toimage(mask, mode="RGBA")
        street_im = scipy.misc.toimage(image)
        street_im.paste(mask, box=None, mask=mask)

        # Add the prediction to the folder
        yield os.path.basename(image_file), np.array(street_im)

def save_inference_samples(runs_dir, data_dir, sess, image_shape,
                           logits, keep_prob, input_image, epoch):
    """
    save model weights and generate samples.
    :param runs_dir: directory where model weights and samples will be saved
    :param data_dir: directory where the Kitty dataset is stored
    :param sess: TF Session
    :param image_shape: shape of the input image for prediction
    :param logits: TF Placeholder for the FCN prediction
    :param keep_prob: TF Placeholder for dropout keep probability
    :param input_image: TF Placeholder for input images
    :param epochs: Number of epochs or Final label
    """
    # Make folder for current run
    output_dir = os.path.join(runs_dir, str(time.time()))
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # Run NN on test images and save them to disk
    print('Epoch {} finished. Saving test images to: {}'.format(epoch, output_dir))
    image_outputs = gen_test_output(sess, logits, keep_prob, input_image, os.path.join(data_dir, 'data_road/testing'), image_shape)

    # Save the image output
    for name, image in image_outputs:
        scipy.misc.imsave(os.path.join(output_dir, name), image)

    # Save the model
    saver = tf.train.Saver()
    filefcn_path = os.path.join(output_dir, 'fcn-{}.ckpt'.format(epoch))
    save_path = saver.save(sess, filefcn_path)
    print('Model saved to: {}'.format(filefcn_path))
