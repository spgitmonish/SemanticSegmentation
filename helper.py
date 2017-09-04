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
from collections import namedtuple
from pathlib import Path
import matplotlib.pyplot as plt

# Flag to dictate whether to use KITTI_SEMANTIC dataset or City Scapes 
# NOTE: 1 represents KITTI_SEMANTIC and 2 represents City Scapes
DATASET_TO_USE = 1
 
# NOTE: A named tuple is a tuple which allows access of it's elements by name
# Named tuple object where each entry has the following:
# 'class': Name of the class
# 'colorname': Name of the color associated with the class
# 'color': Tuple
Label = namedtuple('Label', ['class_name', 'color_value'])

if DATASET_TO_USE == 1:
    # List of valid truth ground labels
    # NOTE: Ground truth classes based on: http://adas.cvc.uab.es/s2uad/?page_id=11
    gt_labels = [
	    Label('Sky',	np.array([128, 128, 128])), # Gray
	    Label('Building', 	np.array([128,   0, 0])),   # Red
	    Label('Road',       np.array([128,  64, 128])), # Pink
	    Label('Sidewalk', 	np.array([0,     0, 192])), # Blue
	    Label('Fence',      np.array([64,   64, 128])), # Grey-purple
	    Label('Vegetation', np.array([128, 128, 0])),   # Dark Yellow
	    Label('Pole', 	np.array([192, 192, 128])), # Light Yellow
	    Label('Car', 	np.array([64,    0, 128])), # Purple
	    Label('Sign', 	np.array([192, 128, 128])), # Salmon
	    Label('Pedestrian', np.array([64,   64, 0])),   # Yellow-Brown
	    Label('Cyclist', 	np.array([0,   128, 192]))  # Light Blue
    ]
elif DATASET_TO_USE == 2:
    # NOTE: Ground truth classes based on: www.cityscapes-dataset.com/downloads/
    gt_labels = [
        Label('Unlabeled',     np.array([0,     0,   0])), # White
        Label('Dynamic',       np.array([111,  74,   0])), # Olive
        Label('Ground',        np.array([ 81,   0,  81])), # Tyrian Purple
        Label('Road',          np.array([128,  64, 128])), # Cannon Pink
        Label('Sidewalk',      np.array([244,  35, 232])), # Shocking Pink
        Label('Parking',       np.array([250, 170, 160])), # Sundown
        Label('Rail track',    np.array([230, 150, 140])), # Tonys Pink
        Label('Building',      np.array([ 70,  70,  70])), # Charcoal
        Label('Wall',          np.array([102, 102, 156])), # Scampi
        Label('Fence',         np.array([190, 153, 153])), # Rosy Brown
        Label('Guard Rail',    np.array([180, 165, 180])), # London Hue
        Label('Bridge',        np.array([150, 100, 100])), # Copper Rose
        Label('Tunnel',        np.array([150, 120,  90])), # Beaver
        Label('Pole',          np.array([153, 153, 153])), # Nobel
        Label('Traffic light', np.array([250, 170,  30])), # Dark Tangerine
        Label('Traffic sign',  np.array([220, 220,   0])), # Chartreuse Yellow
        Label('Vegetation',    np.array([107, 142,  35])), # Olive Drab
        Label('Terrain',       np.array([152, 251, 152])), # Pale Green
        Label('Sky',           np.array([ 70, 130, 180])), # Steel Blue
        Label('Person',        np.array([220,  20,  60])), # Crimson
        Label('Rider',         np.array([255,   0,   0])), # Read
        Label('Car',           np.array([  0,   0, 142])), # Dark Blue
        Label('Truck',         np.array([  0,   0,  70])), # Navy Blue
        Label('Bus',           np.array([  0,  60, 100])), # Prussian Blue
        Label('Caravan',       np.array([  0,   0,  90])), # Navy Blue
        Label('Trailer',       np.array([  0,   0, 110])), # Navy Blue
        Label('Train',         np.array([  0,  80, 100])), # Blue Lagoon
        Label('Motorcycle',    np.array([  0,   0, 230])), # Blue
        Label('Bicycle',       np.array([119,  11,  32]))  # Falu Red
    ]
else:
    raise RuntimeError('Invalid Dataset option: ' + DATASET_TO_USE)

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

def load_kitti_sem_data(data_folder, validation_fraction):
    """
    Load the data and and return the sets which have paths to validation and training
    :param data_dir: Path to folder that contains all the datasets
    :param validation_fraction: Fraction to store for validation
    """
    # Get all the image and label paths
    image_paths = glob(os.path.join(data_folder, 'RGB', '*.png'))
    label_paths = {os.path.basename(path): path for path in glob(os.path.join(data_folder, 'GT', '*.png'))}
    
    # Check if there is an image folder
    num_images = len(image_paths)
    if num_images == 0:
        raise RuntimeError('No data files found in ' + data_dir)

    # Randomly shuffle before setting up the validation and training sets
    random.shuffle(image_paths)
    validation_images = image_paths[:int(validation_fraction*num_images)]
    training_images = image_paths[int(validation_fraction*num_images):]
    
    # Return the list of paths to images for validation and training sets and also the labels path
    return validation_images, training_images, label_paths

def load_city_scapes_data(data_folder):
    """
    Load the data and and return the sets which have paths to validation and training
    :param data_dir: Path to folder that contains all the datasets
    :param validation_fraction: Fraction to store for validation
    """
    # List of paths to the training images and labels
    training_image_paths = glob(os.path.join(data_folder, 'leftImg8bit/train/', '**/*.png'))

    # List of paths to the validation images and labels
    validation_image_paths = glob(os.path.join(data_folder, 'leftImg8bit/val/', '**/*.png'))

    # Return the list of paths to images for validation and training sets and also the labels path
    return validation_image_paths, training_image_paths

def gen_batch_function(image_paths, label_paths, image_shape, data_type=None):
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
                if DATASET_TO_USE == 1:
                    # Path to the corresponding label image
                    gt_image_file = label_paths[os.path.basename(image_file)]
                elif DATASET_TO_USE == 2:
                    # Get parts of the image file path
                    image_file_parts = Path(image_file).parts
                    # Get the splits of the image file path
                    # NOTE: The last 4 parts are specific to the image file i.e. ../leftImg8bit/train/<city>/<filename>
                    image_file_splits = '/'.join(image_file_parts[:-4])
                    # Add the extra './' at the beginning
                    image_file_splits = './' + image_file_splits
                    # Compose the path for accessing the ground truth images for the corresponding city
                    label_path = os.path.join(image_file_splits, ('gtFine' + '/' + data_type + '/'))
                    label_path = os.path.join(label_path, image_file_parts[-2])
                    # Get the image file name and replace it with the corresponding label name
                    image_file_name = os.path.basename(image_file)
                    label_file_name = image_file_name.replace('leftImg8bit', 'gtFine_color')
                    # Path to the corresponding label image
                    gt_image_file = os.path.join(label_path, label_file_name)
                else:  
                    raise RuntimeError('DATASET_TO_USE set inoorrectly before "gen_batches_function" function call')

                # Reshape the images(original and ground truth)
                image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)
                gt_image = scipy.misc.imresize(scipy.misc.imread(gt_image_file), image_shape)
                
                # NOTE: Applicable only for the gtFine dataset
                if DATASET_TO_USE == 2:
                    # Convert the RGBA ground truth image to RGB
                    gt_image = cv2.cvtColor(gt_image, cv2.COLOR_RGBA2RGB)

                # Add histogram equalization along the 'Y' channel of a YUV image
                # NOTE: This is supposed to help remove shadows in the image
                image_yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
                image_yuv[:,:,0] = cv2.equalizeHist(image_yuv[:,:,0])
                image = cv2.cvtColor(image_yuv, cv2.COLOR_YUV2RGB)
                
                # For composing the boolean representation of each ground truth label
                gt_bg = np.zeros([image_shape[0], image_shape[1]], dtype=bool)
                gt_labels_list = []
		
                # Go through all the ground truth label colors and 
                # ground truth images for comparison
                for label in gt_labels[1:]:
                    # Array representing the ground truth in boolean form
                    gt_current = np.all(gt_image == label.color_value, axis=2)
                    gt_bg |= gt_current
                    gt_labels_list.append(gt_current)
					
                # Final composition of the ground truth
                gt_bg = ~gt_bg
                # Stack the ground truth corresponding to the labels along the 3rd dimension
                # NOTE: The *operator is for unpacking 
                gt_all = np.dstack([gt_bg, *gt_labels_list])
                
                # Append the original images and the ground truth labels
                images.append(image)
                gt_images.append(gt_all)

            yield np.array(images), np.array(gt_images)
    return get_batches_fn

def gen_test_kitti_sem_output(sess, logits, keep_prob, image_pl, data_folder, image_shape):
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
    for image_file in glob(os.path.join(data_folder, 'RGB', '*.png')):
        # Resize the image to the input image size
        image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)

        # Perform segmentation to the test image(add a mask reresenting predictions)
        im_softmax = sess.run(tf.nn.softmax(logits), {keep_prob: 1.0, image_pl: [image]})
	
        # Image to apply masking on 
        street_im = scipy.misc.toimage(image)

        # Applying segmentation coloring
        for index in range(len(gt_labels)):
            color_value = gt_labels[index].color_value
            filtered_softmax = im_softmax[:, index].reshape(image_shape[0], image_shape[1])
            segmentation = (filtered_softmax > 0.5).reshape(image_shape[0], image_shape[1], 1)
            mask = np.dot(segmentation, np.array([[color_value[0], color_value[1], color_value[2], 63]]))
            mask = scipy.misc.toimage(mask, mode="RGBA")                
            street_im.paste(mask, box=None, mask=mask)

        # Add the prediction to the folder
        yield os.path.basename(image_file), np.array(street_im)

def gen_test_city_scapes_output(sess, logits, keep_prob, image_pl, data_folder, image_shape):
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
    # Choose 20 random samples for testing
    testing_image_paths = glob(os.path.join(data_folder, 'leftImg8bit/test/', '**/*.png'))
    random.shuffle(testing_image_paths)

    for image_file in testing_image_paths[:20]:
        # Resize the image to the input image size
        image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)

        # Perform segmentation to the test image(add a mask reresenting predictions)
        im_softmax = sess.run(tf.nn.softmax(logits), {keep_prob: 1.0, image_pl: [image]})
	
        # Image to apply masking on 
        street_im = scipy.misc.toimage(image)

        # Applying segmentation coloring
        for index in range(len(gt_labels)):
            color_value = gt_labels[index].color_value
            filtered_softmax = im_softmax[:, index].reshape(image_shape[0], image_shape[1])
            segmentation = (filtered_softmax > 0.5).reshape(image_shape[0], image_shape[1], 1)
            mask = np.dot(segmentation, np.array([[color_value[0], color_value[1], color_value[2], 63]]))
            mask = scipy.misc.toimage(mask, mode="RGBA")                
            street_im.paste(mask, box=None, mask=mask)

        # Add the prediction to the folder
        yield image_file, np.array(street_im)

def save_inference_kitti_sem_samples(runs_dir, data_dir, sess, image_shape,
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
    output_dir = os.path.join(runs_dir, 'KS_' + str(time.time()))
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # Run NN on test images and save them to disk
    print('Epoch {} finished. Saving test images to: {}'.format(epoch, output_dir))
    image_outputs = gen_test_kitti_sem_output(sess, logits, keep_prob, input_image, os.path.join(data_dir, 'KITTI_SEMANTIC/Testing'), image_shape)

    # Save the image output
    for name, image in image_outputs:
        scipy.misc.imsave(os.path.join(output_dir, name), image)
    
    # Save the model
    saver = tf.train.Saver()
    filefcn_path = os.path.join(output_dir, 'fcn-{}.ckpt'.format(epoch))
    save_path = saver.save(sess, filefcn_path)
    print('Model saved to: {}'.format(filefcn_path))

def save_inference_city_scapes_samples(runs_dir, data_dir, sess, image_shape,
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
    output_dir = os.path.join(runs_dir, 'CS_' + str(time.time()))
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # Run NN on test images and save them to disk
    print('Epoch {} finished. Saving test images to: {}'.format(epoch, output_dir))
    image_outputs = gen_test_city_scapes_output(sess, logits, keep_prob, input_image, os.path.join(data_dir, 'KITTI_SEMANTIC/Testing'), image_shape)

    # Save the image output
    for name, image in image_outputs:
        scipy.misc.imsave(os.path.join(output_dir, os.path.basename(name)), image)
        # Get parts of the image file path
        image_file_parts = Path(name).parts
        # Get the splits of the image file path
        # NOTE: The last 4 parts are specific to the image file i.e. ../leftImg8bit/train/<city>/<filename>
        image_file_splits = '/'.join(image_file_parts[:-4])
        # Add the extra './' at the beginning
        image_file_splits = './' + image_file_splits
        # Compose the path for accessing the ground truth images for the corresponding city
        label_path = os.path.join(image_file_splits, ('gtFine' + '/' + data_type + '/'))
        label_path = os.path.join(label_path, image_file_parts[-2])
        # Get the image file name and replace it with the corresponding label name
        image_file_name = os.path.basename(image_file)
        label_file_name = image_file_name.replace('leftImg8bit', 'gtFine_color')
        # Path to the corresponding label image
        gt_image_file = os.path.join(label_path, label_file_name)
        gt_image = scipy.misc.imresize(scipy.misc.imread(gt_image_file), image_shape)
        # Save the label image for comparison
        scipy.misc.imsave(os.path.join(output_dir, os.path.basename(gt_image_file)), gt_image)
    
    # Save the model
    saver = tf.train.Saver()
    filefcn_path = os.path.join(output_dir, 'fcn-{}.ckpt'.format(epoch))
    save_path = saver.save(sess, filefcn_path)
    print('Model saved to: {}'.format(filefcn_path))
