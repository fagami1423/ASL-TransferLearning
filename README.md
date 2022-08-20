# Sign Language Recognition using Keras InceptionV3 (Transfer Learning ) 

This project is a sign language alphabet recognizer using Python, openCV and tensorflow for training InceptionV3 model, a convolutional neural network model for classification of the american sign language.

## Major Goals
- Predict the Sign to text
- Convert text to speech

## Requirements
This project uses python 3.8 and the PIP following packages:
* opencv
* tensorflow
* matplotlib
* numpy

```
pip install -r requirements.txt
```

See requirements.txt and Dockerfile for versions and required APT packages

### Using Docker
```
docker build -t sign-classifier .
docker run -it sign-classifier bash
```
### Install using PIP
```
pip install -r requirements.txt
```
## Training (Inceptionv3)

To train the model, use the following command (see framework github link for more command options):
```
python train.py
```
If you're using the provided dataset, it may take up to three hours.
  
## Classifying using webcam (demo)
  
To test classification, use the following command:
```
python predict.py
```

## Training our own CNN model in Google colab
Model used: Sequential model
Run the jupyter notebook ASl_recognition.ipynb


