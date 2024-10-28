# Import necessary libraries
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import time
# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the dataset with incorrect geometry
# incorrect_data = pd.read_csv(r'C:\Users\Andrew\PycharmProjects\pythonProject\Data\cleaned_mutual_inductance_dataset_9th_sept_correct_geometry.csv')
incorrect_data = pd.read_csv(r'C:\Users\Andrew\PycharmProjects\pythonProject\Data\sorted_valid_samples_20240923.csv')

# Preprocess the data
features = ['Final Mutual Inductance (nH)']
target = ['Trace Width (Rx L mm)', 'Trace Thickness (Rx L mm)', 'Number of Turns (Rx L)',
          'Trace Width (Rx L+C mm)', 'Trace Thickness (Rx L+C mm)', 'Number of Turns (Rx L+C)',
          'Gap Between Segments (mm)', 'Center Area Percentage', 'Capacitor Area Length (mm)',
          'Capacitor Area Width (mm)', 'Rx Coil Area Length (mm)', 'Rx Coil Area Width (mm)',
          'Tx Coil Area Length (mm)', 'Tx Coil Area Width (mm)', 'Final Mutual Inductance (nH)']

# Extract features and target variables
X_incorrect = incorrect_data[features].values
y_incorrect = incorrect_data[target].values

# Normalize the data
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_incorrect_scaled = scaler_X.fit_transform(X_incorrect)
y_incorrect_scaled = scaler_y.fit_transform(y_incorrect)

# Convert to torch tensors
X_incorrect_tensor = torch.tensor(X_incorrect_scaled, dtype=torch.float32).to(device)
y_incorrect_tensor = torch.tensor(y_incorrect_scaled, dtype=torch.float32).to(device)

# Define the Generator model
class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(128),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(256),
            nn.Linear(256, output_dim)
        )
        self.apply(self.weights_init)  # Apply weights initialization

    def forward(self, x):
        return self.model(x)

    @staticmethod
    def weights_init(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)  # Xavier initialization

# Define the Discriminator model
class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        self.apply(self.weights_init)  # Apply weights initialization

    def forward(self, x):
        return self.model(x)

    @staticmethod
    def weights_init(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)  # Xavier initialization

# Initialize models
latent_dim = 100  # Dimension of the noise vector
input_dim = latent_dim + X_incorrect_tensor.shape[1]  # Noise + input features
output_dim = y_incorrect_tensor.shape[1]  # Design parameters output

generator = Generator(input_dim, output_dim).to(device)
discriminator = Discriminator(output_dim + X_incorrect_tensor.shape[1]).to(device)

# Define the loss functions and optimizers
loss_fn = nn.BCELoss()
feature_matching_loss = nn.L1Loss()  # Feature matching loss
optimizer_G = Adam(generator.parameters(), lr=0.0002)
optimizer_D = Adam(discriminator.parameters(), lr=0.0002)

# Learning rate schedulers
scheduler_G = StepLR(optimizer_G, step_size=1000, gamma=0.9)
scheduler_D = StepLR(optimizer_D, step_size=1000, gamma=0.9)

# Training the cGAN
num_epochs = 50000  # Adjust based on your needs
batch_size = 8
# Start time measurement for training
start_time = time.time()
for epoch in range(num_epochs):
    # Train Discriminator
    real_labels = torch.ones(batch_size, 1).to(device)
    fake_labels = torch.zeros(batch_size, 1).to(device)

    # Sample real data
    idx = np.random.randint(0, X_incorrect_tensor.size(0), batch_size)
    real_data = y_incorrect_tensor[idx]
    real_features = X_incorrect_tensor[idx]

    # Generate fake data
    noise = torch.randn(batch_size, latent_dim).to(device)
    generator_input = torch.cat((noise, real_features), dim=1)
    fake_data = generator(generator_input)

    # Discriminator forward pass with real data
    real_input = torch.cat((real_data, real_features), dim=1)
    real_output = discriminator(real_input)
    loss_real = loss_fn(real_output, real_labels)

    # Discriminator forward pass with fake data
    fake_input = torch.cat((fake_data, real_features), dim=1)
    fake_output = discriminator(fake_input.detach())
    loss_fake = loss_fn(fake_output, fake_labels)

    # Discriminator loss and backprop
    d_loss = loss_real + loss_fake
    optimizer_D.zero_grad()
    d_loss.backward()
    optimizer_D.step()

    # Train Generator
    fake_output = discriminator(fake_input)
    g_loss = loss_fn(fake_output, real_labels)

    # Feature matching loss for Generator
    feature_loss = feature_matching_loss(fake_data, real_data)
    total_g_loss = g_loss + feature_loss

    optimizer_G.zero_grad()
    total_g_loss.backward()
    optimizer_G.step()

    # Step the learning rate schedulers
    scheduler_G.step()
    scheduler_D.step()

    if epoch % 100 == 0:
        print(f"Epoch [{epoch}/{num_epochs}] | D Loss: {d_loss.item():.4f} | G Loss: {total_g_loss.item():.4f}")
# End time measurement for training
end_time = time.time()

# Calculate elapsed time in seconds
elapsed_time = end_time - start_time
elapsed_time_minutes = elapsed_time / 60
print(f"Total time required to train the model: {elapsed_time:.2f} seconds ({elapsed_time_minutes:.2f} minutes)")

# Start time measurement for generation
start_generation_time = time.time()
# Generate new design samples
num_samples = 100000
noise = torch.randn(num_samples, latent_dim).to(device)
input_indices = np.random.choice(X_incorrect_tensor.shape[0], num_samples, replace=True)
input_features = X_incorrect_tensor[input_indices]

# Concatenate noise and input features for the generator
generator_input = torch.cat((noise, input_features), dim=1)

# Generate design parameters
generator.eval()
with torch.no_grad():
    generated_params = generator(generator_input)
# End time measurement for generation
end_generation_time = time.time()

# Calculate elapsed time for generation in seconds
generation_time = end_generation_time - start_generation_time
generation_time_minutes = generation_time / 60
print(f"Time required to generate {num_samples} designs: {generation_time:.2f} seconds ({generation_time_minutes:.2f} minutes)")
# Scale back to original values
generated_params_np = generated_params.cpu().numpy()
generated_params_original = scaler_y.inverse_transform(generated_params_np)

# Create DataFrame with generated parameters
df_generated = pd.DataFrame(generated_params_original, columns=target)

# Convert the 'Final Mutual Inductance (H)' column to nH
# df_generated['Final Mutual Inductance (nH)'] = df_generated['Final Mutual Inductance (H)'] * 1e9

# Drop the original 'Final Mutual Inductance (H)' column if not needed
# df_generated.drop('Final Mutual Inductance (H)', axis=1, inplace=True)

# Save the generated WPT parameters to a CSV file
df_generated.to_csv('generated_wpt_parameters_from_valid_samples_20240924_8.csv', index=False)
print("Generated WPT parameters saved to 'generated_wpt_parameters.csv'.")
