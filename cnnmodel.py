import os
import re
import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt

class Model():
    def __init__(self):
        self.base_model = keras.applications.InceptionV3(weights='imagenet', include_top=False)
        self.base_model.trainable = False
            
    def load_images(self,folder_name):
        image_size = (299, 299)
        batch_size = 32

        train_ds = keras.preprocessing.image_dataset_from_directory(
            folder_name,
            validation_split=0.2,
            subset="training",
            seed=1337,
            image_size=image_size,
            batch_size=batch_size,
        )
        val_ds = keras.preprocessing.image_dataset_from_directory(
            folder_name,
            validation_split=0.2,
            subset="validation",
            seed=1337,
            image_size=image_size,
            batch_size=batch_size,
        )
        return train_ds, val_ds

    def get_model(self):
        x = self.base_model.output
        x = keras.layers.GlobalAveragePooling2D()(x)
        x = keras.layers.Dense(1024, activation='relu')(x)
        outputs = keras.layers.Dense(900, activation='softmax')(x)
        model = keras.Model(inputs=self.base_model.input, outputs=outputs)
        model.compile(optimizer = keras.optimizers.SGD(learning_rate=0.0001,momentum=0.9),
                loss = "sparse_categorical_crossentropy",
                metrics = ["accuracy"])
        return model
    
    def save_model(self, model, model_name):
        model.save("model/"+model_name)
        print("Model saved")
    
    def load_model(self,model_name):
        model = keras.models.load_model(model_name)
        return model

    def plot_accuracy_curve(self, history):
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']

        loss = history.history['loss']
        val_loss = history.history['val_loss']

        plt.figure(figsize=(8,8))
        plt.subplot(211)
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.subplot(212)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        plt.savefig('accuracy_curve.png')



    


        


    

    