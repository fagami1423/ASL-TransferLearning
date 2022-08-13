
import matplotlib.pyplot as plt

from cnnmodel import Model

base_model = Model()

#load images from the dataset folder
train_ds, val_ds = base_model.load_images("dataset")

#get the pretrained model from the base model
model = base_model.get_model()

#train the model
model.fit(train_ds, epochs=10, validation_data=val_ds)  

#save the model
base_model.save_model(model,"modelv1.h5")

history = model.evaluate(val_ds)

#plot the accuracy and loss
base_model.plot_accuracy_curve(history)

