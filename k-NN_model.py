# Train Algae Model using CNN with Improved Accuracy

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import image_dataset_from_directory
import numpy as npj
import numpy as np
import cv2


# Path to Dataset
dataset_path = 'C:/Users/abdvk/Downloads/Scope_3_Microalgae_shape_texture_convolution_classification-main/Scope_3_Microalgae_shape_texture_convolution_classification-main/Batch_1'

# Load Dataset
train_dataset = image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(224, 224),
    batch_size=32
)

validation_dataset = image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(224, 224),
    batch_size=32
)

# Data Augmentation
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
    layers.RandomBrightness(0.2)
])

# Model Building with Regularization
model = keras.Sequential([
    layers.Rescaling(1./255, input_shape=(224, 224, 3)),
    data_augmentation,
    layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(256, (3, 3), activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(len(train_dataset.class_names), activation='softmax')
])

# Compile Model with a Lower Learning Rate
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train Model
history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=1,
    class_weight=None # Use class weights if data is imbalanced
)

# Save Model
model.save('algae_species_classifier_improved.keras')



print("Model trained and saved successfully.")

# Predict Species from Image
 # Predict
def predict_species(image_path, model_path=r"C:\Users\abdvk\Downloads\Scope_3_Microalgae_shape_texture_convolution_classification-main\Scope_3_Microalgae_shape_texture_convolution_classification-main\algae_species_classifier_improved.keras"):
    model = keras.models.load_model(model_path)
    class_names = train_dataset.class_names

    # Load and preprocess the image
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)

    # Predict
    prediction = model.predict(image)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    print(f"Predicted Species: {predicted_class} with {confidence:.2f}% confidence.")

# Example Usage
predict_species(r"C:\Users\abdvk\Downloads\Scope_3_Microalgae_shape_texture_convolution_classification-main\Scope_3_Microalgae_shape_texture_convolution_classification-main\Batch_1\Chlamydomonas_Reinhardtii\Chlamy_2.jpg")



   