import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input
from tensorflow.keras.applications.imagenet_utils import decode_predictions

image_size = 224
model = VGG19(weights="imagenet")

camera = cv2.VideoCapture(0)

while camera.isOpened():
    ok, cam_frame = camera.read()

    if not ok:
        break

    frame = cv2.resize(cam_frame, (image_size, image_size))
    numpy_image = img_to_array(frame)
    image_batch = np.expand_dims(numpy_image, axis=0)
    processed_image = preprocess_input(image_batch.copy())

    predictions = model.predict(processed_image, verbose=0)
    label = decode_predictions(predictions, top=1)

    name = label[0][0][1]
    score = label[0][0][2] * 100

    text = "{} {:.1f}%".format(name, score)

    cv2.putText(
        cam_frame,
        "VGG19: " + text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255,0,0),
        2
    )

    cv2.imshow("VGG19 Webcam", cam_frame)

    if cv2.waitKey(30) & 0xFF == 27:
        break

camera.release()
cv2.destroyAllWindows()
