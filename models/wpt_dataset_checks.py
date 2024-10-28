import numpy as np
import pandas as pd
from scipy.special import ellipk, ellipe
import os
import time

# Constants
mu_0 = 4 * np.pi * 1e-7  # Permeability of free space (H/m)

# Define the updated Rx coil dimensions (in mm)
rx_total_width = 26.61  # Total available width for Rx segments
rx_total_length = 33.34  # Total available length for Rx segments

# Define the Tx coil dimensions (5 times larger than Rx)
tx_total_width = rx_total_width * 5  # Total available width for Tx
tx_total_length = rx_total_length * 5  # Total available length for Tx

# Define the range for Trace Width, Thickness, Gap, and Center Area Percentage
trace_widths = np.arange(0.06, 0.82, 0.04)  # Varying from 0.06 mm to 0.8 mm
trace_thicknesses = np.arange(0.06, 0.82, 0.04)  # Varying from 0.06 mm to 0.8 mm
gaps = np.arange(0.05, 1, 0.05)  # Gap between segments varies from 0.1 mm to 0.5 mm
center_area_percentages = np.arange(0.15, 0.30, 0.05)  # Varying center area percentage from 15% to 30%

# Set constant distance between coils (z = 20 mm)
z = 20.0  # Constant distance in mm

# Initialize a counter to track the number of iterations
counter = 0

# Total number of iterations for progress tracking
total_iterations = len(trace_widths) * len(trace_thicknesses) * len(gaps) * len(center_area_percentages)

# Check if there's an existing partial dataset
output_file = 'Data/mutual_inductance_dataset_updated.csv'
if os.path.exists(output_file):
    df_existing = pd.read_csv(output_file)
    counter = len(df_existing)
    dataset = df_existing.to_dict('records')
    print(f"Resuming from {counter} configurations.")
else:
    dataset = []

# Function to calculate mutual inductance between filaments
def calculate_mutual_inductance_for_filaments(a, b, c, d, z):
    R = np.sqrt((a + c) ** 2 + (b + d) ** 2 + z ** 2)
    if R == 0:
        return 0.0

    k_sq = 4 * a * c / ((a + c) ** 2 + (b + d) ** 2 + z ** 2)
    if k_sq < 0 or k_sq > 1:
        return 0.0

    try:
        K_k = ellipk(k_sq)
        E_k = ellipe(k_sq)
    except ValueError:
        return 0.0

    sqrt_ac = np.sqrt(a * c)
    M = (mu_0 / np.pi) * sqrt_ac * ((1 - k_sq / 2) * K_k - E_k) / R

    if np.isnan(M) or np.isinf(M):
        return 0.0

    return M

# Function to calculate the mutual inductance between segments
def calculate_mutual_inductance_for_segment(segment_length_rx, segment_width_rx, segment_length_tx, segment_width_tx,
                                            N_rx, Tw_rx, T_rx, gap_length, z):
    if N_rx <= 0:
        return 0.0

    P_T_rx = max(int(Tw_rx / 0.025), 1)  # Finer division
    Q_T_rx = max(int(T_rx / 0.01), 1)    # Finer division

    M_total = 0.0

    for q_T_rx in range(1, 2 * Q_T_rx + 2):
        for p_T_rx in range(1, 2 * P_T_rx + 2):
            for q_T_tx in range(1, 2 * Q_T_rx + 2):
                for p_T_tx in range(1, 2 * P_T_rx + 2):
                    a_rx = (segment_length_rx - p_T_rx * (Tw_rx + gap_length)) / N_rx
                    b_rx = (segment_width_rx - q_T_rx * (T_rx + gap_length)) / N_rx

                    a_tx = (segment_length_tx - p_T_tx * (Tw_rx + gap_length)) / N_rx
                    b_tx = (segment_width_tx - q_T_tx * (T_rx + gap_length)) / N_rx

                    # Ensure filament dimensions are positive
                    if min(a_rx, b_rx, a_tx, b_tx) <= 0:
                        continue

                    M_ij = calculate_mutual_inductance_for_filaments(a_rx, b_rx, a_tx, b_tx, z)
                    M_total += M_ij

    # Direct summation approach, no normalization factor
    return M_total

# Iterate over all combinations of trace width, thickness, gap, and center area percentage
start_time = time.time()  # Start the timer
for center_area_percentage in center_area_percentages:
    for Tw_rx in trace_widths:
        for T_rx in trace_thicknesses:
            for gap_length in gaps:
                cap_area_length = rx_total_length * center_area_percentage
                cap_area_width = rx_total_width * center_area_percentage

                coil_area_length_rx = rx_total_length - cap_area_length
                coil_area_width_rx = rx_total_width - cap_area_width

                coil_area_length_tx = tx_total_length - (cap_area_length * 5)
                coil_area_width_tx = tx_total_width - (cap_area_width * 5)

                available_length_rx = coil_area_length_rx - gap_length
                available_length_tx = coil_area_length_tx - gap_length

                # Ensure available lengths are positive
                if available_length_rx > 0 and available_length_tx > 0:
                    segment_length_rx_L = available_length_rx / 2
                    segment_length_rx_LC = available_length_rx / 2
                    segment_length_tx = available_length_tx / 2

                    turn_pitch_rx_L = Tw_rx + gap_length
                    N_rx_L = int(segment_length_rx_L / turn_pitch_rx_L)
                    turn_pitch_rx_LC = Tw_rx + gap_length
                    N_rx_LC = int(segment_length_rx_LC / turn_pitch_rx_LC)
                    N_tx = int(segment_length_tx / turn_pitch_rx_L)

                    # Ensure segment lengths and number of turns are positive
                    if segment_length_rx_L > 0 and segment_length_rx_LC > 0 and segment_length_tx > 0 \
                            and N_rx_L > 0 and N_rx_LC > 0 and N_tx > 0:

                        M1_L = calculate_mutual_inductance_for_segment(segment_length_rx_L, coil_area_width_rx,
                                                                       segment_length_tx, coil_area_width_tx, N_rx_L, Tw_rx,
                                                                       T_rx, gap_length, z)
                        M1_LC = calculate_mutual_inductance_for_segment(segment_length_rx_LC, coil_area_width_rx,
                                                                        segment_length_tx, coil_area_width_tx, N_rx_LC, Tw_rx,
                                                                        T_rx, gap_length, z)

                        M_final = M1_L + M1_LC

                        dataset.append({
                            'Trace Width (Rx L mm)': Tw_rx,
                            'Trace Thickness (Rx L mm)': T_rx,
                            'Number of Turns (Rx L)': N_rx_L,
                            'Trace Width (Rx L+C mm)': Tw_rx,
                            'Trace Thickness (Rx L+C mm)': T_rx,
                            'Number of Turns (Rx L+C)': N_rx_LC,
                            'Gap Between Segments (mm)': gap_length,
                            'Center Area Percentage': center_area_percentage,
                            'Capacitor Area Length (mm)': cap_area_length,
                            'Capacitor Area Width (mm)': cap_area_width,
                            'Rx Coil Area Length (mm)': coil_area_length_rx,
                            'Rx Coil Area Width (mm)': coil_area_width_rx,
                            'Tx Coil Area Length (mm)': coil_area_length_tx,
                            'Tx Coil Area Width (mm)': coil_area_width_tx,
                            'Mutual Inductance Segment L (H)': M1_L,
                            'Mutual Inductance Segment L+C (H)': M1_LC,
                            'Final Mutual Inductance (H)': M_final
                        })

                        counter += 1
                        if counter % 100 == 0 or counter == total_iterations:
                            end_time = time.time()  # End the timer
                            elapsed_time = end_time - start_time  # Calculate the elapsed time
                            print(
                                f"Processed {counter}/{total_iterations} configurations. Time for last 100 configurations: {elapsed_time:.2f} seconds.")
                            start_time = time.time()  # Reset the timer

                            # Save progress
                            pd.DataFrame(dataset).to_csv(output_file, index=False)

# Final save to ensure no data is lost
pd.DataFrame(dataset).to_csv(output_file, index=False)
print(f"Final dataset saved with {counter} configurations.")
