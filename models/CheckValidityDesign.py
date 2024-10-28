import numpy as np
import pandas as pd

# Set display options to show all rows and columns
pd.set_option('display.max_rows', None)  # Show all rows
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.width', None)  # Adjust the width to display all columns in a single line
pd.set_option('display.max_colwidth', None)  # Display full content of each cell
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


# Read the DataFrame from CSV - replace 'your_data.csv' with your actual file path
df = pd.read_csv(r'C:\Users\Andrew\PycharmProjects\pythonProject\models\generated_wpt_parameters_from_valid_samples_20240924_8.csv')

# Define the total dimensions for validation
rx_total_width = 26.61  # Total available width for Rx segments
rx_total_length = 33.34  # Total available length for Rx segments

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

# Calculate the percentage of valid designs
total_samples = len(df)
valid_samples = df['Is_Valid'].sum()
valid_percentage = (valid_samples / total_samples) * 100

# Print the results
print(f"Total samples: {total_samples}")
print(f"Number of valid samples: {valid_samples}")
print(f"Percentage of valid samples: {valid_percentage:.2f}%")

# Filter the DataFrame to include only valid rows
valid_df = df[df['Is_Valid'] == True]

# Filter the DataFrame to include only valid rows with at least 5 turns in both columns
filtered_df = valid_df[(valid_df['Number of Turns (Rx L)'] >= 5) & (valid_df['Number of Turns (Rx L+C)'] >= 5)]
print(len(filtered_df))
# Sort the filtered DataFrame based on the 'Final Mutual Inductance (nH)' column in ascending order
sorted_filtered_df = filtered_df.sort_values(by='Final Mutual Inductance (nH)', ascending=False)
# Check if there are any rows that meet the criteria
if not filtered_df.empty:
    # Find the maximum value of 'Final Mutual Inductance (nH)' among the filtered valid rows
    max_final_mutual_inductance = filtered_df['Final Mutual Inductance (nH)'].max()

    # Find the row(s) with the maximum 'Final Mutual Inductance (nH)'
    max_designs = filtered_df[filtered_df['Final Mutual Inductance (nH)'] == max_final_mutual_inductance]

    # Print the maximum value and the corresponding rows
    print(
        f"Maximum value of Final Mutual Inductance (nH) among prioritized designs: {max_final_mutual_inductance:.2f} nH")
    print("Design(s) with the maximum Final Mutual Inductance and prioritized number of turns:")
    print(max_designs)
else:
    print("No designs with the specified number of turns found.")

print(sorted_filtered_df.head())

# Define the function to calculate Tx parameters based on Rx parameters
import pandas as pd


# Define the function to systematically derive the number of turns for Tx
def calculate_tx_turns(row):
    # Extract necessary values from the DataFrame
    trace_width_L = row['Trace Width (Rx L mm)']
    trace_width_LC = row['Trace Width (Rx L+C mm)']
    gap_between_segments = row['Gap Between Segments (mm)']
    tx_length_L = row['Tx Coil Area Length (mm)'] / 2  # Assuming equal division
    tx_length_LC = row['Tx Coil Area Length (mm)'] / 2  # Assuming equal division

    # Calculate Turn Pitch for Tx segments
    turn_pitch_L = trace_width_L + gap_between_segments
    turn_pitch_LC = trace_width_LC + gap_between_segments

    # Calculate the number of turns for Tx segments
    N_tx_L = tx_length_L / turn_pitch_L
    N_tx_LC = tx_length_LC / turn_pitch_LC

    # Floor the values to ensure integer turns
    N_tx_L = int(N_tx_L)
    N_tx_LC = int(N_tx_LC)

    return N_tx_L, N_tx_LC


# Apply the function to the DataFrame and add columns for the number of turns
df[['Number of Turns (Tx L)', 'Number of Turns (Tx L+C)']] = df.apply(
    lambda row: calculate_tx_turns(row), axis=1, result_type='expand')

# Display the updated DataFrame
print(df.head())

# Sample DataFrame creation (replace this with your actual DataFrame)
df = pd.DataFrame({
    'Trace Width (Rx L mm)': [0.127659],
    'Trace Thickness (Rx L mm)': [0.145178],
    'Number of Turns (Rx L)': [18.384783],
    'Trace Width (Rx L+C mm)': [0.127235],
    'Trace Thickness (Rx L+C mm)': [0.154748],
    'Number of Turns (Rx L+C)': [17.835093],
    'Gap Between Segments (mm)': [0.605948],
    'Center Area Percentage': [0.149752],
    'Capacitor Area Length (mm)': [5.001089],
    'Capacitor Area Width (mm)': [3.991521],
    'Rx Coil Area Length (mm)': [28.339254],
    'Rx Coil Area Width (mm)': [22.618044],
    'Tx Coil Area Length (mm)': [141.69507],
    'Tx Coil Area Width (mm)': [113.092735],
    'Final Mutual Inductance (nH)': [171.106],
    'Is_Valid': [True],
    'Number of Turns (Tx L)': [96],
    'Number of Turns (Tx L+C)': [96]
})

import pandas as pd

# Sample DataFrame with example values (replace with actual DataFrame)
df = pd.DataFrame({
    'Trace Width (Rx L mm)': [0.127659],
    'Trace Thickness (Rx L mm)': [0.145178],
    'Number of Turns (Tx L)': [96],
    'Trace Width (Rx L+C mm)': [0.127235],
    'Trace Thickness (Rx L+C mm)': [0.154748],
    'Number of Turns (Tx L+C)': [96],
    'Gap Between Segments (mm)': [0.605948],
    'Tx Coil Area Length (mm)': [141.69507],
    'Tx Coil Area Width (mm)': [113.092735]
})


import pandas as pd

# Sample DataFrame with example values (replace with your actual DataFrame)
df = pd.DataFrame({
    'Trace Width (Rx L mm)': [0.127659],
    'Trace Thickness (Rx L mm)': [0.145178],
    'Number of Turns (Tx L)': [96],
    'Trace Width (Rx L+C mm)': [0.127235],
    'Trace Thickness (Rx L+C mm)': [0.154748],
    'Number of Turns (Tx L+C)': [96],
    'Gap Between Segments (mm)': [0.605948],
    'Tx Coil Area Length (mm)': [141.69507],
    'Tx Coil Area Width (mm)': [113.092735]
})

def verify_tx_geometrical_validity(row):
    # Extract parameters for Tx coil
    trace_width_L = row['Trace Width (Rx L mm)']
    trace_thickness_L = row['Trace Thickness (Rx L mm)']
    num_turns_L = row['Number of Turns (Tx L)']

    trace_width_LC = row['Trace Width (Rx L+C mm)']
    trace_thickness_LC = row['Trace Thickness (Rx L+C mm)']
    num_turns_LC = row['Number of Turns (Tx L+C)']

    gap_between_segments = row['Gap Between Segments (mm)']
    tx_length = row['Tx Coil Area Length (mm)']
    tx_width = row['Tx Coil Area Width (mm)']

    # Calculate required length and width for Tx L segment
    required_length_L = num_turns_L * (trace_width_L + gap_between_segments)
    required_width_L = num_turns_L * (trace_thickness_L + gap_between_segments)

    # Calculate required length and width for Tx L+C segment
    required_length_LC = num_turns_LC * (trace_width_LC + gap_between_segments)
    required_width_LC = num_turns_LC * (trace_thickness_LC + gap_between_segments)

    # Total required length and width
    total_required_length = required_length_L + required_length_LC
    total_required_width = required_width_L + required_width_LC

    # Validation check: Do the required dimensions fit within the available Tx coil area?
    is_valid_length = total_required_length <= tx_length
    is_valid_width = total_required_width <= tx_width

    # Return overall validity
    return is_valid_length and is_valid_width

# Apply the validation function to the DataFrame
df['Tx_Is_Geometrically_Valid'] = df.apply(verify_tx_geometrical_validity, axis=1)

# Display the DataFrame with validity results
print(df[['Trace Width (Rx L mm)', 'Trace Thickness (Rx L mm)', 'Number of Turns (Tx L)',
         'Trace Width (Rx L+C mm)', 'Trace Thickness (Rx L+C mm)', 'Number of Turns (Tx L+C)',
         'Tx Coil Area Length (mm)', 'Tx Coil Area Width (mm)', 'Tx_Is_Geometrically_Valid']])


import pandas as pd

# Sample DataFrame based on the provided values
data = {
    'Trace Width (Rx L mm)': [0.152522],
    'Trace Thickness (Rx L mm)': [0.395155],
    'Number of Turns (Rx L)': [5.546375],
    'Trace Width (Rx L+C mm)': [0.151654],
    'Trace Thickness (Rx L+C mm)': [0.385073],
    'Number of Turns (Rx L+C)': [5.019623],
    'Gap Between Segments (mm)': [1.244952],
    'Rx Coil Area Length (mm)': [28.340910],
    'Rx Coil Area Width (mm)': [22.618511],
    'Capacitor Area Length (mm)': [5.001623],
    'Capacitor Area Width (mm)': [3.992626]
}

df = sorted_filtered_df

def calculate_distance_between_turns(row):
    # Maximum dimensions for the coil
    max_width = 26.61  # Maximum allowed width of the coil
    max_length = 33.34  # Maximum allowed length of the coil

    # Capacitor area dimensions
    capacitor_width = row['Capacitor Area Width (mm)']
    capacitor_length = row['Capacitor Area Length (mm)']

    # Available space after accounting for the capacitor area
    available_width = max_width - capacitor_width
    available_length = max_length - capacitor_length

    # Extract values for Segment 1
    trace_width_1 = row['Trace Width (Rx L mm)']
    num_turns_1 = row['Number of Turns (Rx L)']
    gap_segment = row['Gap Between Segments (mm)']

    # Calculate total trace width and gap for Segment 1
    total_trace_width_1 = num_turns_1 * trace_width_1
    total_gap_width_1 = (num_turns_1 - 1) * gap_segment

    # Check if available space is sufficient for Segment 1
    available_space_1 = (available_width / 2) - (total_trace_width_1 + total_gap_width_1)
    if available_space_1 < 0:
        print("Warning: Insufficient space for Segment 1. Adjust parameters.")
        distance_segment_1 = float('nan')  # Assign NaN or an appropriate value to indicate an issue
    else:
        # Calculate distance between turns for Segment 1
        distance_segment_1 = available_space_1 / num_turns_1

    # Extract values for Segment 2
    trace_width_2 = row['Trace Width (Rx L+C mm)']
    num_turns_2 = row['Number of Turns (Rx L+C)']

    # Calculate total trace width and gap for Segment 2
    total_trace_width_2 = num_turns_2 * trace_width_2
    total_gap_width_2 = (num_turns_2 - 1) * gap_segment

    # Check if available space is sufficient for Segment 2
    available_space_2 = (available_width / 2) - (total_trace_width_2 + total_gap_width_2)
    if available_space_2 < 0:
        print("Warning: Insufficient space for Segment 2. Adjust parameters.")
        distance_segment_2 = float('nan')  # Assign NaN or an appropriate value to indicate an issue
    else:
        # Calculate distance between turns for Segment 2
        distance_segment_2 = available_space_2 / num_turns_2

    return distance_segment_1, distance_segment_2

# Apply the function to each row in the dataframe and expand the results into two columns
df[['Distance Between Turns (Segment 1)', 'Distance Between Turns (Segment 2)']] = df.apply(
    lambda row: calculate_distance_between_turns(row), axis=1, result_type='expand'
)

print(df.head())

