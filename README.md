# Image Captioning using "Show, Attend and Tell"
This repository contains the implementation of an Image Captioning model based on the paper "Show, Attend and Tell" by Kelvin Xu et al. The model is trained to generate captions for given images using an attention-based encoder-decoder network.

# About the Model
"Show, Attend and Tell" is a neural network model for image captioning that uses an attention-based encoder-decoder architecture. The model first encodes the image using a Convolutional Neural Network (CNN), which is followed by a recurrent neural network (RNN) that generates the caption word by word. Additionally, the model uses an attention mechanism that allows it to focus on different parts of the image at different time steps, resulting in more accurate and detailed captions.

# Dataset
The model is trained on the MS-COCO (Microsoft Common Objects in Context) dataset, which consists of images and corresponding captions. The dataset contains over 330,000 images with 5 captions each, resulting in a total of 1.6 million captions. We have used a preprocessed version of the dataset that has been converted into numerical representations.

# Dependencies
The code is implemented in Python 3.7 using the following libraries:

PyTorch 1.9.0
torchvision 0.10.0
NumPy 1.19.5
NLTK 3.6.3
Pillow 8.3.2

# Training the Model
To train the model, run the following command:

`python train.py`
This will start the training process and save the model weights after each epoch in the models folder.

# Generating Captions
To generate captions for new images, run the following command:

python generate.py --image_path <path_to_image_file>
Replace <path_to_image_file> with the path to the image file for which you want to generate the caption. The script will preprocess the image, feed it to the trained model, and generate a caption for the image.

Acknowledgments
We would like to thank Kelvin Xu et al. for their paper "Show, Attend and Tell" and the MS-COCO dataset for providing the data for this project. We have also used some code snippets from the PyTorch tutorial on Image Captioning.
