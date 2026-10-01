# Example 4.16 - AutoEncoder for MNIST

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, Flatten, Reshape, Input, InputLayer
from tensorflow.keras.models import Sequential, Model

# Load MNIST dataset
(x_train, _), (x_test, _) = mnist.load_data()

# Normalize data
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

print(x_train.shape)
print(x_test.shape)


# Build AutoEncoder
def build_autoencoder(img_shape, code_size):

    encoder = Sequential([
        InputLayer(shape=img_shape),
        Flatten(),
        Dense(code_size)
    ])

    decoder = Sequential([
        InputLayer(shape=(code_size,)),
        Dense(int(np.prod(img_shape))),
        Reshape(img_shape)
    ])

    return encoder, decoder


# Create encoder and decoder
IMG_SHAPE = x_train[0].shape

encoder, decoder = build_autoencoder(IMG_SHAPE, 32)


# Build AutoEncoder model
inp = Input(shape=IMG_SHAPE)

code = encoder(inp)

reconstruction = decoder(code)

autoencoder = Model(inp, reconstruction)


# Compile model
autoencoder.compile(
    optimizer='adamax',
    loss='mse'
)


# Show model summary
print(autoencoder.summary())


# Train model
history = autoencoder.fit(
    x=x_train,
    y=x_train,
    epochs=20,
    validation_data=(x_test, x_test)
)


# Plot training loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()


# Show image
def show_image(x):
    plt.imshow(
        np.clip(x, 0, 1),
        cmap='gray'
    )


# Visualize AutoEncoder result
def visualize(img, encoder, decoder):

    code = encoder.predict(
        img[None],
        verbose=0
    )[0]

    reco = decoder.predict(
        code[None],
        verbose=0
    )[0]

    plt.figure(figsize=(9, 3))

    # Original image
    plt.subplot(1, 3, 1)
    plt.title('Original')
    show_image(img)

    # Encoded image
    plt.subplot(1, 3, 2)
    plt.title('Code')

    plt.imshow(
        code.reshape(
            [code.shape[-1] // 2, -1]
        ),
        cmap='gray'
    )

    # Reconstructed image
    plt.subplot(1, 3, 3)
    plt.title('Reconstructed')
    show_image(reco)

    plt.show()


# Test 5 images
for i in range(5):
    visualize(
        x_test[i],
        encoder,
        decoder
    )