
import matplotlib.pyplot as plt

from Model import Model


base_model = Model()

#load images from the dataset folder
train_ds, val_ds = base_model.load_images("dataset")

#get the pretrained model from the base model
model = base_model.get_model()

#train the model
model.fit(train_ds, epochs=10, validation_data=val_ds)  

#save the model
model.save("model/modelv1.h5")

history = model.evaluate(val_ds)


#plot the accuracy and loss
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()
plt.savefig("accuracy_loss.png")