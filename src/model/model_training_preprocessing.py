import tensorflow as tf
from tensorflow.keras import layers
import os
import numpy as np

class model_training_preprocessing:
    def __init__(self, train_dir, val_dir, test_dir, batch_size=16, img_height=299, img_width=299):
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.test_dir = test_dir
        self.batch_size = batch_size
        self.img_height = img_height
        self.img_width = img_width

    def load_data(self):
        # Load data and create data generators
        train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            self.train_dir,
            labels='inferred',
            label_mode='int',
            image_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            shuffle=True,
            seed=1
        )
        print(f"Number of training batches:   {train_ds.cardinality()}")
        print(f"Total training samples:       {train_ds.cardinality().numpy() * self.batch_size}")

        val_ds = tf.keras.preprocessing.image_dataset_from_directory(
            self.val_dir,
            labels='inferred',
            label_mode='int',
            image_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            shuffle=True,
            seed=1
        )
        print(f"Number of validation batches: {val_ds.cardinality()}")
        print(f"Total validation samples:     {val_ds.cardinality().numpy() * self.batch_size}")

        test_ds = tf.keras.preprocessing.image_dataset_from_directory(
            self.test_dir,
            labels='inferred',
            label_mode='int',
            image_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            shuffle=True,
            seed=1
        )
        print(f"Number of test batches:       {test_ds.cardinality()}")
        print(f"Total test samples:           {test_ds.cardinality().numpy() * self.batch_size}")

        return train_ds, val_ds, test_ds

    def preprocess_data(self, train_ds, val_ds, test_ds):
        # Preprocessing pipeline
        resize_and_rescale = tf.keras.Sequential([
            layers.Resizing(self.img_height, self.img_width),
            layers.Rescaling(1./255)
        ])

        augmentation = tf.keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.2),
            layers.RandomZoom(0.2),
        ])

        # Apply augmentation + preprocessing to training set
        self.train_ds = train_ds.map(lambda x, y: (resize_and_rescale(augmentation(x)), y))
        # Apply only preprocessing to val/test sets
        self.val_ds = val_ds.map(lambda x, y: (resize_and_rescale(x), y))
        self.test_ds = test_ds.map(lambda x, y: (resize_and_rescale(x), y))

        return self.train_ds, self.val_ds, self.test_ds
