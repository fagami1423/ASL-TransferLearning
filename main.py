import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import Model


#load model
model = keras.models.load_model("model/modelv1.h5")

#load images from the dataset folder
train_ds, val_ds = Model.load_images("dataset")

#prediction on single image
predictions = model.predict(val_ds)