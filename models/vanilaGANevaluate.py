import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from models.GAN2 import generator, df_normalized, device, scaler

# Set the generator to evaluation mode
generator.eval()

# Generate random noise
latent_dim = 100
num_samples = 10000  # Number of new design samples to generate
noise = torch.randn(num_samples, latent_dim).to(device)  # Generate noise for multiple design samples

# Use existing mutual inductance values as input features (or you could use random inputs)
input_data = df_normalized[['Mutual Inductance Segment L (H)', 'Mutual Inductance Segment L+C (H)']].values

# Randomly sample num_samples rows with replacement
indices = np.random.choice(input_data.shape[0], num_samples, replace=True)
input_features = input_data[indices]

input_features = torch.tensor(input_features, dtype=torch.float32).to(device)

# Concatenate noise and input features
generator_input = torch.cat((noise, input_features), dim=1)

# Generate the design parameters
with torch.no_grad():  # Disable gradient calculation for evaluation
    generated_params = generator(generator_input)

# Convert generated_params back to the original scale
placeholder = np.zeros((generated_params.shape[0], scaler.scale_.shape[0]))
placeholder[:, :generated_params.shape[1]] = generated_params.cpu().numpy()
original_scale_params = scaler.inverse_transform(placeholder)
final_generated_params = original_scale_params[:, :generated_params.shape[1]]

# Define the parameter names (including 'Final Mutual Inductance (H)')
parameter_names = [
    'Trace Width (Rx L mm)', 'Trace Thickness (Rx L mm)', 'Number of Turns (Rx L)',
    'Trace Width (Rx L+C mm)', 'Trace Thickness (Rx L+C mm)', 'Number of Turns (Rx L+C)',
    'Gap Between Segments (mm)', 'Center Area Percentage', 'Capacitor Area Length (mm)',
    'Capacitor Area Width (mm)', 'Rx Coil Area Length (mm)', 'Rx Coil Area Width (mm)',
    'Tx Coil Area Length (mm)', 'Tx Coil Area Width (mm)', 'Final Mutual Inductance (nH)'
]

# Create DataFrame for generated data
df_generated_params = pd.DataFrame(final_generated_params, columns=parameter_names)

# **Correctly Extract and Assign the Mutual Inductance Column**
df_generated_params['Mutual Inductance'] = generated_params[:, -1].cpu().numpy()  # Ensure correct extraction

# Round the relevant columns to two decimal places
for col in parameter_names[:-1]:  # Skip the 'Final Mutual Inductance (H)' column
    df_generated_params[col] = df_generated_params[col].round(2)


# print(df_generated_params['Final Mutual Inductance (nH)'])

# Convert 'Final Mutual Inductance (H)' to nanoHenries (nH) and round
df_generated_params['Final Mutual Inductance (nH)'] = (df_generated_params['Final Mutual Inductance (nH)'] * 1e9).round(2)

# Convert 'Mutual Inductance' to nanoHenries (nH) and round
df_generated_params['Mutual Inductance'] = (df_generated_params['Mutual Inductance'] * 1e9).round(2)

# Find the optimal design parameters that yield the highest mutual inductance
optimal_design = df_generated_params.loc[df_generated_params['Mutual Inductance'].idxmax()]
# Exclude 'Mutual Inductance' from the optimal design parameters before displaying
optimal_design = optimal_design.drop('Mutual Inductance')
# Display the optimal design parameters
print("\nOptimal Design Parameters:")
print(optimal_design)

# Plot the distribution of mutual inductance
plt.figure(figsize=(12, 6))
plt.hist(df_generated_params['Mutual Inductance'], bins=30, alpha=0.5, label='Generated Mutual Inductance (nH)', color='red')
plt.title('Distribution of Generated Mutual Inductance Values')
plt.xlabel('Mutual Inductance (nH)')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True)
plt.show()
