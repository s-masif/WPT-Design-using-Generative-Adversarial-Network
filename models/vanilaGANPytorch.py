import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os
import torch
torch.cuda.empty_cache()
print(torch.cuda.is_available())  # Should return True if CUDA is available
# Check if GPU is available and set device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
# Load and preprocess the data
df = pd.read_csv('Data/mutual_inductance_dataset_updated_2.0.csv')

# Normalize the data
scaler = MinMaxScaler()
df_normalized = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

# Split the data into input and output
input_features = df_normalized[
    ['Mutual Inductance Segment L (H)', 'Mutual Inductance Segment L+C (H)', 'Final Mutual Inductance (H)']].values
output_features = df_normalized.drop(columns=['Mutual Inductance Segment L (H)', 'Mutual Inductance Segment L+C (H)',
                                              'Final Mutual Inductance (H)']).values


# Define the Generator network
class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
        )

    def forward(self, x):
        return self.model(x)


# Define the Discriminator network
class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


# Set up the GAN
latent_dim = 100  # dimension of the random noise vector
generator = Generator(latent_dim + input_features.shape[1], output_features.shape[1]).to(device)
discriminator = Discriminator(output_features.shape[1] + input_features.shape[1]).to(device)  # Adjust input dim to match conditional input

# Optimizers and Loss Function
optimizer_G = optim.Adam(generator.parameters(), lr=0.0002)
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002)
criterion = nn.BCELoss()

# Convert data to tensors and move to device
real_data = torch.tensor(output_features, dtype=torch.float32).to(device)
condition = torch.tensor(input_features, dtype=torch.float32).to(device)
real_labels = torch.ones(real_data.size(0), 1).to(device)
fake_labels = torch.zeros(real_data.size(0), 1).to(device)

# Training the GAN
def train_gan(num_epochs=50000, save_interval=1000, early_stopping_patience=500):
    best_loss_G = float('inf')
    patience_counter = 0

    for epoch in range(num_epochs):
        # Train Discriminator
        noise = torch.randn(real_data.size(0), latent_dim).to(device)
        fake_data = generator(torch.cat((noise, condition), dim=1))

        optimizer_D.zero_grad()
        output_real = discriminator(torch.cat((real_data, condition), dim=1))
        output_fake = discriminator(torch.cat((fake_data.detach(), condition), dim=1))
        loss_D = criterion(output_real, real_labels) + criterion(output_fake, fake_labels)
        loss_D.backward()
        optimizer_D.step()

        # Train Generator
        optimizer_G.zero_grad()
        output_fake = discriminator(torch.cat((fake_data, condition), dim=1))
        loss_G = criterion(output_fake, real_labels)
        loss_G.backward()
        optimizer_G.step()

        if epoch % save_interval == 0:
            print(f"Epoch {epoch}/{num_epochs}, Loss D: {loss_D.item()}, Loss G: {loss_G.item()}")

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
    torch.save(generator.state_dict(), 'models/generator.pth')
    torch.save(discriminator.state_dict(), 'models/discriminator.pth')
    print("Models saved.")

if __name__ == "__main__":
    train_gan()
