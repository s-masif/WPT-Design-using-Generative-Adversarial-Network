import pandas as pd
import numpy as np

# Define the validation function
def validate_geometric_constraints_full(N_L, T_L, G_L, N_LC, T_LC, G_LC, W_center, W_total, L_center, L_total,
                                        G_segments):
    """
    Validates whether the total width and length required by segments L and L+C fit within the total available width and length.

    Parameters:
        N_L (float): Number of turns in segment L (will be floored to ensure it's a whole number).
        T_L (float): Trace thickness for segment L.
        G_L (float): Gap between turns for segment L.
        N_LC (float): Number of turns in segment L+C (will be floored to ensure it's a whole number).
        T_LC (float): Trace thickness for segment L+C.
        G_LC (float): Gap between turns for segment L+C.
        W_center (float): Center area width.
        W_total (float): Total available width for the coil design.
        L_center (float): Center area length.
        L_total (float): Total available length for the coil design.
        G_segments (float): Gap between segment L and segment L+C.

    Returns:
        bool: True if the design fits within the available width and length and has valid positive values, False otherwise.
    """
    # Floor the number of turns to ensure they are whole numbers
    N_L = np.floor(N_L)
    N_LC = np.floor(N_LC)

    # Check for negative values; if any parameter is negative, the design is invalid
    if N_L < 0 or N_LC < 0 or T_L < 0 or G_L < 0 or T_LC < 0 or G_LC < 0 or G_segments < 0:
        return False

    # Calculate required width and length
    width_L = N_L * (T_L + G_L)
    width_LC = N_LC * (T_LC + G_LC)
    total_required_width = width_L + G_segments + width_LC + W_center

    length_L = N_L * (T_L + G_L)
    length_LC = N_LC * (T_LC + G_LC)
    total_required_length = length_L + G_segments + length_LC + L_center

    # Validate both width and length
    return total_required_width <= W_total and total_required_length <= L_total


# Define the total dimensions for validation
rx_total_width = 26.61  # Total available width for Rx segments
rx_total_length = 33.34  # Total available length for Rx segments


# Function to validate and filter valid samples from a dataframe
def validate_dataframe(df):
    # Apply the validation function to each row and create a new column 'Is_Valid'
    df['Is_Valid'] = df.apply(lambda row: validate_geometric_constraints_full(
        N_L=row['Number of Turns (Rx L)'],
        T_L=row['Trace Thickness (Rx L mm)'],
        G_L=row['Trace Width (Rx L mm)'],
        N_LC=row['Number of Turns (Rx L+C)'],
        T_LC=row['Trace Thickness (Rx L+C mm)'],
        G_LC=row['Trace Width (Rx L+C mm)'],
        W_center=row['Capacitor Area Width (mm)'],
        W_total=rx_total_width,
        L_center=row['Capacitor Area Length (mm)'],
        L_total=rx_total_length,
        G_segments=row['Gap Between Segments (mm)']
    ), axis=1)

    # Return only valid rows
    return df[df['Is_Valid'] == True]


# Load your actual dataframes from CSV files (replace with your actual file paths)
df_10k = pd.read_csv(r'C:\Users\Andrew\PycharmProjects\pythonProject\Result\generated_wpt_parameters_20240920_1.csv')  # Replace with actual path
df_20k = pd.read_csv(r'C:\Users\Andrew\PycharmProjects\pythonProject\Result\generated_wpt_parameters_20240920_2.csv')  # Replace with actual path
df_30k = pd.read_csv(r'C:\Users\Andrew\PycharmProjects\pythonProject\Result\generated_wpt_parameters_20240920_3.csv')  # Replace with actual path
df_40k = pd.read_csv(r'C:\Users\Andrew\PycharmProjects\pythonProject\Result\generated_wpt_parameters_20240920_4.csv')  # Replace with actual path
df_50k = pd.read_csv(r'C:\Users\Andrew\PycharmProjects\pythonProject\Result\generated_wpt_parameters_20240920_5.csv')  # Replace with actual path

# Validate each dataframe and get only valid samples
valid_10k = validate_dataframe(df_10k)
valid_20k = validate_dataframe(df_20k)
valid_30k = validate_dataframe(df_30k)
valid_40k = validate_dataframe(df_40k)
valid_50k = validate_dataframe(df_50k)

# Combine all valid samples into one dataframe
all_valid_samples = pd.concat([valid_10k, valid_20k, valid_30k, valid_40k, valid_50k])

# Sort the combined valid samples by 'Final Mutual Inductance (nH)' in descending order
sorted_valid_samples = all_valid_samples.sort_values(by='Final Mutual Inductance (nH)', ascending=False)

# Save the sorted valid samples to a CSV file
sorted_valid_samples.to_csv('sorted_valid_samples_20240923.csv', index=False)

# Print the first few rows of the sorted dataframe
print(f"Total valid samples: {len(sorted_valid_samples)}")
print(sorted_valid_samples.head())
