class model_training_tuning:
    def __init__(self, train_ds, val_ds, test_ds, img_height=299, img_width=299):
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.test_ds = test_ds
        self.img_height = img_height
        self.img_width = img_width
        self.model = None
        self.history = None
        self.epochs = 10


    def build_model(self):
        # Load the InceptionV3 model with pre-trained weights
        base_model = tf.keras.applications.InceptionV3(
            include_top=False,
            weights='imagenet',
            input_shape=(self.img_height, self.img_width, 3)
        )

        # Freeze the base model
        base_model.trainable = False

        # Create a new model on top of the base model
        inputs = tf.keras.Input(shape=(self.img_height, self.img_width, 3))
        x = base_model(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(1024, activation='relu')(x)
        outputs = layers.Dense(1, activation='sigmoid')(x)

        self.model = tf.keras.Model(inputs, outputs)
        self.model.compile(optimizer='adam',
                           loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                           metrics=['accuracy'])    
        
        return self.model
    
    def train_model(self):
        # Train the model
        self.history = self.model.fit(
            self.train_ds,
            validation_data=self.val_ds,
            epochs=self.epochs,
            batch_size=16
        )
        return self.history
    
    def evaluate_model(self):
        # Evaluate the model on the test set
        test_loss, test_acc = self.model.evaluate(self.test_ds)
        print(f"Test accuracy: {test_acc:.4f}")
        return test_loss, test_acc