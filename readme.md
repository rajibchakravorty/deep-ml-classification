### Deep Learning for classification of CIFAR-10 dataset
Note: This is not aimed to achieive best performance on CIFAR-10 dataset. This is rather aimed as an end-to-end code to run image classification problem with deep CNN other than CIFAR-10. Hope is that with minor changes this can be made ready to be run on other image classification tasks.

#### CIFAR-10 Data
Python pickle binary can be downloaded from [here](https://www.cs.toronto.edu/~kriz/cifar.html).
Note: There are build-in function in almost every library these days to read the binary file and extract the images as `numpy` array (cifar_helper/data2image.py). In this code though, I transferred these arrays into images and saved those in separate folders marked my class labels. This additional step is to ensure that the code can be used for other classification problem where images will be stored on HDD.

#### Libraries
Scikit-Image, Scikit-Learn, Lasage, Theano

#### Environment
Python, Miniconda

#### Image Object
In order to make sure that the data can be loaded on machines with memory constraints, the images are represented as a `list` of ImageObject (ml_utility/image_object.py). The ImageObject stores the path of the image file and label associated with the image (as a number). Moreover, it also has a `transformer` object. The `transformer` (an instance of `Transformer` class; ml_utility/transformer.py) object transforms the image. This is best suited for creating data augmentation.

See `prepare_data/prepare_imo.py` for how to use ImageObject to create the list including data augmentation.

#### Experiment
The `experiment` directory contains the code to run the classification experiment.

##### config.py
Holds some constant required by the training/testing scheme. For example, Hyperparameters, Previously stored model, File names to save results after every epoch etc.

##### train.py
1. Loads the list of ImageObjects; Loads the train/validation dataset
2. Loads the weights from a pre-stored file (if provided)
3. Initiate some thread to read the images from the files
4. Runs the epochs; After every epoch reports the performance on training data and validation data
5. Stores the best model so far

##### cnn_X.py
The definition of deep CNN architecture## Deep Learning for classification of CIFAR-10 dataset

##### test.py
Code to run the trained network on the trained weights

#### Highlights

1. Achieves 79.1% accuracy on the test data-this is very close to the validation performance
2. Reads the images in parallel through multiple reader threads; NOTE: However, the training update and reading are sequential
