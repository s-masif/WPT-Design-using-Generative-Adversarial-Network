import tensorflow as tf
from tensorflow.keras import layers, models
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import os
# Create compatibility for deprecated aliases
# Compatibility fix for deprecated numpy aliases

# Check if GPU is available and set device accordingly
device = "/GPU:0" if tf.config.list_physical_devices('GPU') else "/CPU:0"
print(device)

# Load and preprocess the data
df = pd.read_csv(r'C:\Users\Andrew\PycharmProjects\pythonProject\Data\mutual_inductance_dataset_updated_2.0.csv')

# Normalize the data
scaler = MinMaxScaler()
df_normalized = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
print(df_normalized.columns)
# Debugging: Print column names and lengths
for col in df_normalized.columns:
    print(f"'{col}'", len(col))

# Check if required columns exist and are accessible
try:
    col1 = df_normalized['Mutual Inductance Segment L (H)']
    col2 = df_normalized['Mutual Inductance Segment L+C (H)']
    col3 = df_normalized['Final Mutual Inductance (H)']
    input_features = np.column_stack((col1, col2, col3))
    print("Successfully extracted input features.")
except KeyError as e:
    print(f"Column not found: {e}")
except Exception as e:
    print(f"An error occurred: {e}")
# Split the data into input and output
col1 = df_normalized['Mutual Inductance Segment L (H)'].values
col2 = df_normalized['Mutual Inductance Segment L+C (H)'].values
col3 = df_normalized['Final Mutual Inductance (H)'].values

input_features = np.column_stack((col1, col2, col3))

# input_features = df_normalized[['Mutual Inductance Segment L (H)', 'Mutual Inductance Segment L+C (H)', 'Final Mutual Inductance (H)']].values
# output_features = df_normalized.drop(columns=['Mutual Inductance Segment L (H)', 'Mutual Inductance Segment L+C (H)', 'Final Mutual Inductance (H)']).values
columns_to_keep = [col for col in df_normalized.columns if col not in ['Mutual Inductance Segment L (H)', 'Mutual Inductance Segment L+C (H)', 'Final Mutual Inductance (H)']]
print("Columns to keep:", columns_to_keep)
try:
    output_features = np.column_stack([df_normalized[col].values for col in columns_to_keep])
    print(f"Successfully stacked columns into output_features with shape {output_features.shape}.")
except Exception as e:
    print(f"An error occurred while stacking columns: {e}")
# Check if all columns in columns_to_keep are valid
# output_features = None
# try:
#     for col in columns_to_keep:
#         print(f"Accessing column: {col}")
#         data = df_normalized[col].values
#         print(f"Column {col} accessed successfully with shape {data.shape}.")
#
#     output_features = df_normalized[columns_to_keep].values
#     print("Successfully selected all columns.")
# except Exception as e:
#     print(f"An error occurred while accessing columns: {e}")
# Define the Generator network
def build_generator(input_dim, output_dim):
    model = models.Sequential()
    model.add(layers.Dense(128, input_dim=input_dim))
    model.add(layers.ReLU())
    model.add(layers.Dense(256))
    model.add(layers.ReLU())
    model.add(layers.Dense(512))
    model.add(layers.ReLU())
    model.add(layers.Dense(output_dim))
    return model

# Define the Discriminator network
def build_discriminator(input_dim):
    model = models.Sequential()
    model.add(layers.Dense(512, input_dim=input_dim))
    model.add(layers.LeakyReLU(0.2))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(256))
    model.add(layers.LeakyReLU(0.2))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(128))
    model.add(layers.LeakyReLU(0.2))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(1, activation='sigmoid'))
    return model

# Set up the GAN
latent_dim = 100  # dimension of the random noise vector
generator = build_generator(latent_dim + input_features.shape[1], output_features.shape[1])
discriminator = build_discriminator(output_features.shape[1] + input_features.shape[1])

# Compile the discriminator
discriminator.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002), loss='binary_crossentropy')

# Build and compile the GAN
discriminator.trainable = False
gan_input = layers.Input(shape=(latent_dim + input_features.shape[1],))
gen_output = generator(gan_input)
gan_output = discriminator(layers.concatenate([gen_output, gan_input[:, -input_features.shape[1]:]]))
gan = models.Model(gan_input, gan_output)
gan.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002), loss='binary_crossentropy')

# Convert data to tensors and move to device
real_data = tf.convert_to_tensor(output_features, dtype=tf.float32)
condition = tf.convert_to_tensor(input_features, dtype=tf.float32)
real_labels = tf.ones((real_data.shape[0], 1))
fake_labels = tf.zeros((real_data.shape[0], 1))

# Training the GAN
def train_gan(num_epochs=10000, save_interval=1000, early_stopping_patience=500):
    best_loss_G = float('inf')
    patience_counter = 0

    for epoch in range(num_epochs):
        with tf.device(device):
            # Train Discriminator
            noise = tf.random.normal((real_data.shape[0], latent_dim))
            fake_data = generator(tf.concat([noise, condition], axis=1))

            d_loss_real = discriminator.train_on_batch(tf.concat([real_data, condition], axis=1), real_labels)
            d_loss_fake = discriminator.train_on_batch(tf.concat([fake_data, condition], axis=1), fake_labels)
            loss_D = d_loss_real + d_loss_fake

            # Train Generator
            noise = tf.random.normal((real_data.shape[0], latent_dim))
            loss_G = gan.train_on_batch(tf.concat([noise, condition], axis=1), real_labels)

            if epoch % save_interval == 0:
                print(f"Epoch {epoch}/{num_epochs}, Loss D: {loss_D}, Loss G: {loss_G}")

                # Early stopping check
                if loss_G < best_loss_G:
                    best_loss_G = loss_G
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping at epoch {epoch}.")
                    break

    # Save the trained models
    os.makedirs("models", exist_ok=True)
    generator.save('models/generator.h5')
    discriminator.save('models/discriminator.h5')
    print("Models saved.")

if __name__ == "__main__":
    train_gan()
