import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os

# Check if GPU is available and set device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Load the data
df = pd.read_csv('Data/mutual_inductance_dataset_updated_2.0.csv')
df = df.drop('Number of Turns per Tx Segment', axis=1)


# # Define the minimum values for all 16 output parameters
# min_values_tensor = torch.tensor([
#     0.06,   # Trace Width (Rx L mm)
#     0.06,   # Trace Thickness (Rx L mm)
#     float('-inf'),  # Number of Turns (Rx L) - no lower constraint provided
#     0.06,   # Trace Width (Rx L+C mm)
#     0.06,   # Trace Thickness (Rx L+C mm)
#     float('-inf'),  # Number of Turns (Rx L+C) - no lower constraint provided
#     0.05,   # Gap Between Segments (mm)
#     0.15,   # Center Area Percentage
#     0.0,    # Capacitor Area Length (mm) - cannot be zero or negative
#     0.0,    # Capacitor Area Width (mm) - cannot be zero or negative
#     0.0,    # Rx Coil Area Length (mm) - cannot be zero or negative
#     0.0,    # Rx Coil Area Width (mm) - cannot be zero or negative
#     0.0,    # Tx Coil Area Length (mm) - cannot be zero or negative
#     0.0,    # Tx Coil Area Width (mm) - cannot be zero or negative
#     float('-inf'),  # Number of Turns per Tx Segment - no lower constraint provided
#     float('-inf')   # Final Mutual Inductance (H) - no lower constraint provided
# ], device=device)
#
# # Define the maximum values for all 16 output parameters
# max_values_tensor = torch.tensor([
#     0.8,   # Trace Width (Rx L mm)
#     0.8,   # Trace Thickness (Rx L mm)
#     float('inf'),  # Number of Turns (Rx L) - no upper constraint provided
#     0.8,   # Trace Width (Rx L+C mm)
#     0.8,   # Trace Thickness (Rx L+C mm)
#     float('inf'),  # Number of Turns (Rx L+C) - no upper constraint provided
#     0.95,  # Gap Between Segments (mm)
#     0.3,   # Center Area Percentage
#     float('inf'),  # Capacitor Area Length (mm) - no upper constraint
#     float('inf'),  # Capacitor Area Width (mm) - no upper constraint
#     float('inf'),  # Rx Coil Area Length (mm) - no upper constraint
#     float('inf'),  # Rx Coil Area Width (mm) - no upper constraint
#     float('inf'),  # Tx Coil Area Length (mm) - no upper constraint
#     float('inf'),  # Tx Coil Area Width (mm) - no upper constraint
#     float('inf'),  # Number of Turns per Tx Segment - no upper constraint provided
#     float('inf')   # Final Mutual Inductance (H) - no upper constraint provided
# ], device=device)

# Normalize the cleaned data
scaler = MinMaxScaler()
df_normalized = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

# Split the data into input and output
# Input features (conditioned on 'Final Mutual Inductance (H)')
input_features = df_normalized[['Final Mutual Inductance (H)']].values

# Output features (design parameters)
output_features = df_normalized[['Trace Width (Rx L mm)', 'Trace Thickness (Rx L mm)', 'Number of Turns (Rx L)',
                                 'Trace Width (Rx L+C mm)', 'Trace Thickness (Rx L+C mm)', 'Number of Turns (Rx L+C)',
                                 'Gap Between Segments (mm)', 'Center Area Percentage']].values

# Continue with the rest of your code...
# Define the Generator and Discriminator models, and the GAN training loop as before.

# Define the Generator network
class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, output_dim),
        )

    def forward(self, x):
        return self.model(x)


# Define the Discriminator network
class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()  # Ensure this is present
        )

    def forward(self, x):
        return self.model(x)


# Set up the GAN
latent_dim = 100  # dimension of the random noise vector
generator = Generator(latent_dim + input_features.shape[1], output_features.shape[1]).to(device)
discriminator = Discriminator(output_features.shape[1] + input_features.shape[1]).to(
    device)  # Adjust input dim to match conditional input

# Optimizers and Loss Function
optimizer_G = optim.Adam(generator.parameters(), lr=0.00001, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

criterion = nn.BCELoss()

# Convert data to tensors and move to device
real_data = torch.tensor(output_features, dtype=torch.float32).to(device)
condition = torch.tensor(input_features, dtype=torch.float32).to(device)
real_labels = torch.ones(real_data.size(0), 1).to(device)
fake_labels = torch.zeros(real_data.size(0), 1).to(device)

# Training the GAN
def train_gan(num_epochs=1000, save_interval=1000, early_stopping_patience=500):
    best_loss_G = float('inf')
    patience_counter = 0

    # Initialize a variable to store the best design parameters
    best_design = None
    best_mutual_inductance = float('-inf')

    # A list to store all generated designs and their mutual inductances
    generated_designs = []

    # Lists to store loss history for plotting
    loss_D_history = []
    loss_G_history = []

    for epoch in range(num_epochs):
        # Train Discriminator
        noise = torch.randn(real_data.size(0), latent_dim).to(device)
        fake_data = generator(torch.cat((noise, condition), dim=1))

        # Round the "Number of Turns" to the nearest integer
        fake_data[:, 2] = torch.round(fake_data[:, 2])  # Number of Turns (Rx L)
        fake_data[:, 5] = torch.round(fake_data[:, 5])  # Number of Turns (Rx L+C)

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

        # Store loss history
        loss_D_history.append(loss_D.item())
        loss_G_history.append(loss_G.item())

        # Store the generated design along with its final mutual inductance
        for i in range(fake_data.size(0)):
            generated_designs.append((fake_data[i].cpu().detach().numpy(), condition[i].item()))  # Store parameters and mutual inductance

        # Periodically print losses and save models
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

    # After training, find the design with the highest mutual inductance
    for design, mutual_inductance in generated_designs:
        if mutual_inductance > best_mutual_inductance:
            best_mutual_inductance = mutual_inductance
            best_design = design

    # Print the optimal design parameters
    if best_design is not None:
        print("Optimal design parameters with the highest mutual inductance:")
        print(f"Trace Width (Rx L mm): {best_design[0]}")
        print(f"Trace Thickness (Rx L mm): {best_design[1]}")
        print(f"Number of Turns (Rx L): {best_design[2]}")
        print(f"Trace Width (Rx L+C mm): {best_design[3]}")
        print(f"Trace Thickness (Rx L+C mm): {best_design[4]}")
        print(f"Number of Turns (Rx L+C): {best_design[5]}")
        print(f"Gap Between Segments (mm): {best_design[6]}")
        print(f"Center Area Percentage: {best_design[7]}")
        print(f"Final Mutual Inductance (H): {best_mutual_inductance}")

    # Save the optimal design parameters to a file
    design_file_path = "optimal_design_parameters.csv"
    with open(design_file_path, "w") as f:
        f.write("Trace Width (Rx L mm),Trace Thickness (Rx L mm),Number of Turns (Rx L),Trace Width (Rx L+C mm),Trace Thickness (Rx L+C mm),Number of Turns (Rx L+C),Gap Between Segments (mm),Center Area Percentage,Final Mutual Inductance (H)\n")
        f.write(f"{best_design[0]},{best_design[1]},{best_design[2]},{best_design[3]},{best_design[4]},{best_design[5]},{best_design[6]},{best_design[7]},{best_mutual_inductance}\n")
    print(f"Optimal design parameters saved to {design_file_path}")

    # Save the loss history to a file for plotting later
    loss_file_path = "gan_loss_history.csv"
    with open(loss_file_path, "w") as f:
        f.write("Epoch,Loss D,Loss G\n")
        for epoch in range(len(loss_D_history)):
            f.write(f"{epoch},{loss_D_history[epoch]},{loss_G_history[epoch]}\n")
    print(f"Loss history saved to {loss_file_path}")

    # Save the trained models
    os.makedirs("models", exist_ok=True)
    torch.save(generator.state_dict(), 'models/generator.pth')
    torch.save(discriminator.state_dict(), 'models/discriminator.pth')
    print("Models saved.")



if __name__ == "__main__":
    train_gan()
