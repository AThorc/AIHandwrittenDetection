import tensorflow as tf
from tensorflow.keras.layers import Dense, Reshape, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import mnist  # Esempio con il dataset MNIST

# Caricamento del dataset MNIST
(x_train, _), (_, _) = mnist.load_data()

# Normalizzazione dei dati
x_train = x_train.astype('float32') / 255.0

# Definizione del generatore
generator = Sequential([
    Dense(256, input_shape=(100,), activation='relu'),
    Dense(512, activation='relu'),
    Dense(28*28, activation='sigmoid'),
    Reshape((28, 28))
])

# Definizione del discriminatore
discriminator = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(512, activation='relu'),
    Dense(256, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compilazione del generatore
generator.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002))

# Compilazione del discriminatore
discriminator.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002), metrics=['accuracy'])

# Congelamento del discriminatore durante l'addestramento del generatore
discriminator.trainable = False

# Creazione del modello GAN
gan_input = tf.keras.Input(shape=(100,))
x = generator(gan_input)
gan_output = discriminator(x)
gan = tf.keras.Model(gan_input, gan_output)
gan.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002))

# Addestramento della GAN
batch_size = 32
epochs = 10000

for epoch in range(epochs):
    # Addestramento del discriminatore
    noise = tf.random.normal((batch_size, 100))
    generated_images = generator.predict(noise)
    real_images = x_train[np.random.randint(0, x_train.shape[0], batch_size)]
    combined_images = np.concatenate([generated_images, real_images])
    labels = np.concatenate([np.zeros((batch_size, 1)), np.ones((batch_size, 1))])
    labels += 0.05 * np.random.random(labels.shape)
    d_loss = discriminator.train_on_batch(combined_images, labels)

    # Addestramento del generatore tramite la GAN
    noise = tf.random.normal((batch_size, 100))
    misleading_labels = np.ones((batch_size, 1))
    g_loss = gan.train_on_batch(noise, misleading_labels)

    # Stampa della perdita di addestramento
    print(f'Epoch {epoch}, Discriminator Loss: {d_loss}, Generator Loss: {g_loss}')

# Generazione di nuove immagini
noise = tf.random.normal((10, 100))
generated_images = generator.predict(noise)

# Visualizzazione delle immagini generate
for image in generated_images:
    plt.imshow(image, cmap='gray')
    plt.show()
