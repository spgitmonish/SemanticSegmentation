'''
Since the City Scapes dataset is huge(3475 testing and validation), this script randomly subsamples each of the folders in the testing and validation sets to create a new set of samples with 900 training  and 
100 validation samples in the set

The main intention of this script is to have manageable runtime and still view results without waiting for a long time to get the results, especially if the person doesn't own a GPU and needs to rent space on AWS

NOTE: Before running the script make sure the following is downloaded from https://www.cityscapes-dataset.com/downloads/
1. gtFine_trainvaltest.zip & leftImg8bit_trainvaltest.zip
2. Both the folders are extracted and renamed as 'gtFine' and 'leftImg8bit' respectively
3. Both the folders after extraction and renaming are placed under ./data/CityScapes/
'''
import os
from glob import glob
from pathlib import Path
import random
import numpy as np
import shutil

def run():
	# Reference to the data directory
	data_dir = './data'	
	# Reference to the CityScapes directory
	city_scapes_dir = os.path.join(data_dir, 'CityScapes')
    
    # Throw an exception if the ./data/CityScapes/gtFine or ./data/CityScapes/leftImg8bit folders don't exist
	if Path(os.path.join(city_scapes_dir, 'gtFine')).exists() and Path(os.path.join(city_scapes_dir, 'leftImg8bit')).exists():
		print("Success! Begining to create downsized dataset...")
		# New path for creating the downsized dataset
		new_downsized_dir = os.path.join(data_dir, 'SmallCityScapes')
		
		# Check if the downsized folder exists already
		if os.path.exists(new_downsized_dir):
			print("Downsized folder already created!")		
			return
		else:
			# Make new directory for the downsized data 		
			os.makedirs(new_downsized_dir)
			
			# Original Training data folder
			original_training_folder = os.path.join(city_scapes_dir, 'leftImg8bit/train/')

			# List all the directories under the training folder and make the train and val folders
			# For both the training images and the labels
			for directory in os.listdir(original_training_folder):
				# New training folders
				os.makedirs(os.path.join(new_downsized_dir, 'leftImg8bit/train/' + directory))
				os.makedirs(os.path.join(new_downsized_dir, 'gtFine/train/' + directory))

			# List of paths to the training images
			training_image_paths = glob(os.path.join(original_training_folder, '**/*.png'))

			# Randomly shuffle before copying images
			random.shuffle(training_image_paths)
			
			# Generate a list of 900 unique numbers in range(1, len(training_image_paths)
			rand_train_index_list = random.sample(range(0, len(training_image_paths)), 900)

			# Pick random training images till the list of count 900 is exhausted
			for rand_train_index in rand_train_index_list:
				# Training image file path
				image_file_path = training_image_paths[rand_train_index]

				# Downsize image file path for training
				downsize_image_file_path = image_file_path.replace('CityScapes', 'SmallCityScapes')
				shutil.copy(image_file_path, downsize_image_file_path)

				# Training label file path
				label_dir_path = os.path.dirname(image_file_path).replace('leftImg8bit', 'gtFine')
				label_file_name = os.path.basename(image_file_path).replace('leftImg8bit', 'gtFine_color')
				label_file_path = os.path.join(label_dir_path, label_file_name)

				# Downsize label file path
				downsize_label_file_path = label_file_path.replace('CityScapes', 'SmallCityScapes')
				shutil.copy(label_file_path, downsize_label_file_path)	
			
			# Original Validation data folder
			original_validation_folder = os.path.join(city_scapes_dir, 'leftImg8bit/val/')

			# List of paths to the validation images
			validation_image_paths = glob(os.path.join(original_validation_folder, '**/*.png'))

			# List all the directories under the training folder and make the train and val folders
			# For both the training images and the labels
			for directory in os.listdir(original_validation_folder):
				# New validation folders
				os.makedirs(os.path.join(new_downsized_dir, 'leftImg8bit/val/' + directory))
				os.makedirs(os.path.join(new_downsized_dir, 'gtFine/val/' + directory))

			# Randomly shuffle before copying images
			random.shuffle(validation_image_paths)

			# Generate a list of 100 unique numbers in range(1, len(validation_image_paths)
			rand_valid_index_index_list = random.sample(range(0, len(validation_image_paths)), 100)

			# Pick random training images till the list of count 900 is exhausted
			for rand_valid_index in rand_valid_index_index_list:
				# Validation image file path
				image_file_path = validation_image_paths[rand_valid_index]

				# Downsize image file path for validation
				downsize_image_file_path = image_file_path.replace('CityScapes', 'SmallCityScapes')
				shutil.copy(image_file_path, downsize_image_file_path)

				# Validation label file path
				label_dir_path = os.path.dirname(image_file_path).replace('leftImg8bit', 'gtFine')
				label_file_name = os.path.basename(image_file_path).replace('leftImg8bit', 'gtFine_color')
				label_file_path = os.path.join(label_dir_path, label_file_name)

				# Downsize label file path
				downsize_label_file_path = label_file_path.replace('CityScapes', 'SmallCityScapes')
				shutil.copy(label_file_path, downsize_label_file_path)

		print("Downsize complete!")
	else:
		raise RuntimeError("The necessary folders don't exist, please follow instructions in the header of this file ")

if __name__ == '__main__':
    run()
