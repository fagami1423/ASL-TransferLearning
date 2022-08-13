import os
import re
import tensorflow as tf
import tensorflow.keras as keras

class Model():
    def __init__(self):
        self.base_model = tf.keras.applications.InceptionV3(weights='imagenet', include_top=False)
        self.base_model.trainable = False
            
    def load_images(folder_name):
        image_size = (299, 299)
        batch_size = 32

        train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            folder_name,
            validation_split=0.2,
            subset="training",
            seed=1337,
            image_size=image_size,
            batch_size=batch_size,
        )
        val_ds = tf.keras.preprocessing.image_dataset_from_directory(
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
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(1024, activation='relu')(x)
        outputs = tf.keras.layers.Dense(900, activation='softmax')(x)
        model = tf.keras.Model(inputs=self.base_model.input, outputs=outputs)
        model.compile(optimizer = tf.keras.optimizers.SGD(lr=0.0001,momentum=0.9),
                loss = "sparse_categorical_crossentropy",
                metrics = ["accuracy"])
        return model
    
    def save_model(self, model, model_name):
        model.save("model/"+model_name)
        print("Model saved")
    
    def load_model(self,model_name):
        model = tf.keras.models.load_model(model_name)
        return model
    
    
    


        


    

    