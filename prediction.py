 # Predict
def predict_species(image_path, model_path='algae_species_classifier_improved.h5'):
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

