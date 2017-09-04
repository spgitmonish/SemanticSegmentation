import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests
from helper import *

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # Tag for loading the downloaded and saved VGG-16 ConvNet
    vgg_tag = 'vgg16'

    # Load the model and weights(using the session and tag)
    # from the given path(vgg_path)
    vgg_model = tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)

    # These tensors names are used for creating a FCN(Fully Convolutional Network)
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    # Use the default graph for this thread(should be related to the model loaded)
    vgg_graph = tf.get_default_graph()

    # Input tensor
    vgg_input_tensor = vgg_graph.get_tensor_by_name(vgg_input_tensor_name)
    # Dropout layer tensor
    vgg_keep_prob_tensor = vgg_graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    # Output of pool layer 3
    vgg_layer3_out_tensor = vgg_graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    # Output of pool layer 4
    vgg_layer4_out_tensor = vgg_graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    # Output of layer 7
    vgg_layer7_out_tensor = vgg_graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

    # Return a tuple of tensors from this function
    return vgg_input_tensor, vgg_keep_prob_tensor, vgg_layer3_out_tensor, vgg_layer4_out_tensor, vgg_layer7_out_tensor
tests.test_load_vgg(load_vgg, tf)

def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # Freeze training of layers in VGG-16
    '''vgg_layer3_out = tf.stop_gradient(vgg_layer3_out)
    vgg_layer4_out = tf.stop_gradient(vgg_layer4_out)
    vgg_layer7_out = tf.stop_gradient(vgg_layer7_out)'''

    # Build FCN-8 decoder using upsampling and adding skip connections
    # First make sure that the output shape is same(apply 1x1 convolution)
    vgg_layer7_logits = tf.layers.conv2d(vgg_layer7_out, num_classes, kernel_size=1, name='vgg_layer7_logits')
    vgg_layer4_logits = tf.layers.conv2d(vgg_layer4_out, num_classes, kernel_size=1, name='vgg_layer4_logits')
    vgg_layer3_logits = tf.layers.conv2d(vgg_layer3_out, num_classes, kernel_size=1, name='vgg_layer3_logits')

    # NOTE: The factor of upsampling is equal to the stride of transposed convolution.
    #       The kernel size of the upsampling operation is determined by the identity:
    #       2 * factor - factor % 2.
    # Upsample the output of 1x1 convolution output(start of the decoding process)
    fcn_1 = tf.layers.conv2d_transpose(vgg_layer7_logits, num_classes, kernel_size=4, strides=(2, 2), padding='same', name='fcn_1')

    # Add skip connection from the output of pool layer 4 of VGG
    fcn_2 = tf.add(fcn_1, vgg_layer4_logits, name='fcn_2')

    # Upsample output from fcn_2
    fcn_3 = tf.layers.conv2d_transpose(fcn_2, num_classes, kernel_size=4, strides=(2, 2), padding='same', name='fcn_3')

    # Add skip connection from the output of pool layer 3 of VGG
    fcn_4 = tf.add(fcn_3, vgg_layer3_logits, name='fcn_2')

    # Final decoder output after last upsampling
    fcn_output = tf.layers.conv2d_transpose(fcn_4, num_classes, kernel_size=16, strides=(8, 8), padding='same', name='fcn_output')

    # Return the output tensor(decoder output)
    return fcn_output
tests.test_layers(layers)

def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # Reshape the 4D output and label tensors to 2D:
    # Each row represents a pixel and each column a class.
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    y = tf.reshape(correct_label, (-1, num_classes))

    # Define a loss function(softmax with logits and cross entropy)
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))

    # Use an optimizer for reducing loss(Using Adam optimizer)
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy_loss)

    # Return the result
    return logits, optimizer, cross_entropy_loss
tests.test_optimize(optimize)

def train_nn(sess, epochs, batch_size,
             get_validation_batches_fn, get_training_batches_fn, train_op,
             cross_entropy_loss, input_image, correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_validation_batches_fn: Function to get batches of validation data.
    :param get_training_batches_fn: Function to get batches of training data.
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    # Initialize all the global variables
    sess.run(tf.global_variables_initializer())

    # Run for a certain number of epochs
    for epoch in range(epochs):
        print("Epoch: ", epoch)
        t = time.time()

        # Train on batches
        training_loss = 0
        training_samples = 0
        for X, y in get_training_batches_fn(batch_size):
            training_samples += len(X)
            loss, _ = sess.run([cross_entropy_loss, train_op], feed_dict={input_image: X, correct_label: y, keep_prob: 0.5})
            training_loss += loss

        # Calculate training loss
        training_loss /= training_samples

        # Validation on batches
        validation_loss = 0
        validation_samples = 0
        for X, y in get_validation_batches_fn(batch_size):
            validation_samples += len(X)
            loss = sess.run(cross_entropy_loss, feed_dict={input_image: X, correct_label: y, keep_prob: 1.0})
            validation_loss += loss

        # Calculate training loss
        validation_loss /= validation_samples

        # Print out the stats
        print("Training loss: {}".format(training_loss) + " Validation loss: {}".format(validation_loss))

        # Print time taken
        print("Time: %.3f seconds" % (time.time() - t))
tests.test_train_nn(train_nn)

def run():
    num_classes = len(gt_labels)
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    
    # NOTE: This test applies only for the kitti dataset under data_road(not applicable to kitti_sem branch)
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    # Hyperparameters for training
    if DATASET_TO_USE == 1:
        epochs = 100
        batch_size = 10
    elif DATASET_TO_USE == 2:
        epochs = 1
        batch_size = 10

    lr = 0.0001
    learning_rate = tf.constant(lr)

    # Download the VGG-16 model if it doesn't exist
    helper.maybe_download_pretrained_vgg(data_dir)

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')

        # Seperate the training image set into training and validation sets
        if DATASET_TO_USE == 1:
            validation_path, training_path, label_path = load_kitti_sem_data(os.path.join(data_dir, 'KITTI_SEMANTIC/Training'), 0.1)

            # Create function to get batches for validation and training
            get_validation_batches_fn = helper.gen_batch_function(validation_path, label_path, image_shape, None)
            get_training_batches_fn = helper.gen_batch_function(training_path, label_path, image_shape, None)
        elif DATASET_TO_USE == 2:
            validation_path, training_path = load_city_scapes_data(os.path.join(data_dir, 'CityScapes'))

            # Create function to get batches for validation and training
            get_validation_batches_fn = helper.gen_batch_function(validation_path, None, image_shape, 'val')
            get_training_batches_fn = helper.gen_batch_function(training_path, None, image_shape, 'train')
        else:
            raise RuntimeError('DATASET_TO_USE set incorrectly for "load_data" function call')

        # Build NN using load_vgg, layers, and optimize function
        # Placeholder for model training(batch size, shape[0], shape[1], num_classes)
        correct_label = tf.placeholder(tf.float32, [None, image_shape[0], image_shape[1], num_classes])

        # Get the VGG-16 layers
        vgg_input, keep_prob, vgg_layer3, vgg_layer4, vgg_layer7 = load_vgg(sess, vgg_path)

        # Get the last layer(output) of the network
        fcn_output = layers(vgg_layer3, vgg_layer4, vgg_layer7, num_classes)

        # Get the logits, optimizer and cross entropy loss
        logits, optimizer, cross_entropy_loss = optimize(fcn_output, correct_label, learning_rate, num_classes)

        # Train NN using the train_nn function
        train_nn(sess, epochs, batch_size,
                 get_validation_batches_fn, get_training_batches_fn, optimizer,
                 cross_entropy_loss, vgg_input, correct_label, keep_prob, lr)

        # Save the inference data from the run
        if DATASET_TO_USE == 1:
            save_inference_kitti_sem_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, vgg_input, 'FINAL')
        elif DATASET_TO_USE == 2:
            save_inference_city_scapes_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, vgg_input, 'FINAL')
        # OPTIONAL: Apply the trained model to a video

if __name__ == '__main__':
    run()
