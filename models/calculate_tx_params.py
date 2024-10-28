import math


def calculate_tx_parameters(rx_params, scaling_factor=5):
    """
    Calculate Tx parameters based on Rx parameters and a scaling factor.

    Args:
    - rx_params (dict): Dictionary of Rx coil parameters.
      Keys should include:
        'Trace Width_1', 'Trace Thickness_1', 'Number of Turns_1',
        'Trace Width_2', 'Trace Thickness_2', 'Number of Turns_2',
        'Segment gap', 'Center Percentage', 'Center Length', 'Center Width',
        'Mutual Inductance (nH)'.
    - scaling_factor (float): Factor by which Tx is larger than Rx.

    Returns:
    - tx_params (dict): Dictionary of calculated Tx coil parameters.
    """

    # Extract Rx parameters
    trace_width_rx = rx_params['Trace Width_1']  # Assume similar width for Tx
    trace_thickness_rx = rx_params['Trace Thickness_1']  # Assume similar thickness for Tx
    gap = rx_params['Segment gap']
    center_length = rx_params['Center Length']
    center_width = rx_params['Center Width']

    # Scale Rx dimensions to get Tx dimensions
    tx_length = center_length * scaling_factor
    tx_width = center_width * scaling_factor

    # Calculate the available length for turns
    available_length_tx = tx_length - gap
    available_width_tx = tx_width - gap

    # Determine the number of turns based on the turn pitch
    turn_pitch_tx = trace_width_rx + gap  # Similar pitch for Tx
    number_of_turns_tx = int(available_length_tx / turn_pitch_tx)

    # Ensure a reasonable number of turns (must be positive)
    if number_of_turns_tx <= 0:
        number_of_turns_tx = 1

    # Return Tx parameters
    tx_params = {
        'Trace Width (Tx mm)': trace_width_rx,  # Keeping similar to Rx
        'Trace Thickness (Tx mm)': trace_thickness_rx,  # Keeping similar to Rx
        'Number of Turns (Tx)': number_of_turns_tx,
        'Tx Coil Area Length (mm)': tx_length,
        'Tx Coil Area Width (mm)': tx_width
    }

    return tx_params


# Example Rx parameters
rx_parameters = {
    'Trace Width_1': 0.15,
    'Trace Thickness_1': 0.4,
    'Number of Turns_1': 5,
    'Trace Width_2': 0.15,
    'Trace Thickness_2': 0.4,
    'Number of Turns_2': 5,
    'Segment gap': 0.5,
    'Center Percentage': 0.2,
    'Center Length': 33.34,
    'Center Width': 26.61,
    'Mutual Inductance (nH)': 935.95
}

# Calculate Tx parameters based on Rx parameters
tx_parameters = calculate_tx_parameters(rx_parameters, scaling_factor=5)
print(tx_parameters)



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

df = pd.DataFrame(data)

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

    # Calculate distance between turns for Segment 1
    distance_segment_1 = (available_width / 2 - (total_trace_width_1 + total_gap_width_1)) / num_turns_1

    # Extract values for Segment 2
    trace_width_2 = row['Trace Width (Rx L+C mm)']
    num_turns_2 = row['Number of Turns (Rx L+C)']

    # Calculate total trace width and gap for Segment 2
    total_trace_width_2 = num_turns_2 * trace_width_2
    total_gap_width_2 = (num_turns_2 - 1) * gap_segment

    # Calculate distance between turns for Segment 2
    distance_segment_2 = (available_width / 2 - (total_trace_width_2 + total_gap_width_2)) / num_turns_2

    return distance_segment_1, distance_segment_2

# Apply the function to each row in the dataframe and expand the results into two columns
df[['Distance Between Turns (Segment 1)', 'Distance Between Turns (Segment 2)']] = df.apply(
    lambda row: calculate_distance_between_turns(row), axis=1, result_type='expand'
)

print(df)
