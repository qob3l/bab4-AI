import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input
from tensorflow.keras.applications.imagenet_utils import decode_predictions

model = VGG19(weights="imagenet")

img_path = "foto 1.jpeg"

img = image.load_img(img_path, target_size=(224,224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

predictions = model.predict(x, verbose=0)
results = decode_predictions(predictions, top=5)

print("Hasil prediksi:")
for result in results[0]:
    print(result)
