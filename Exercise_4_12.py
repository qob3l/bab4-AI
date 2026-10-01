from tensorflow.keras.applications import DenseNet201

model = DenseNet201(weights="imagenet")
model.summary()
