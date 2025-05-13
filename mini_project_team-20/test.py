import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, accuracy_score

# 1. Load your model
model = tf.keras.models.load_model("/content/fusion_model (1).h5")

# 2. Prepare the test data
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    '/content/PlantDoc-Dataset/test',         # e.g., 'data/test'
    target_size=(224, 224),        # Match your model input
    batch_size=32,
    class_mode='categorical',      # 'binary' if only 2 classes
    shuffle=False
)

# 3. Predict
predictions = model.predict(test_generator)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = test_generator.classes
class_labels = list(test_generator.class_indices.keys())

# 4. Evaluation Metrics
print("Accuracy:", accuracy_score(true_classes, predicted_classes))

print("\nClassification Report:")
print(classification_report(true_classes, predicted_classes, target_names=class_labels))
