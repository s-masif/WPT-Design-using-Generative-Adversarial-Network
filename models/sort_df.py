import pandas as pd
df = pd.read_csv(r'C:\Users\Andrew\PycharmProjects\pythonProject\models\generated_wpt_parameters_from_valid_samples_20240924_7.csv')
# Assuming df is your DataFrame
# Sort the DataFrame based on the 'Final Mutual Inductance (nH)' column in descending order
df_sorted = df.sort_values(by='Final Mutual Inductance (nH)', ascending=False)

# Save the sorted DataFrame to a CSV file
df_sorted.to_csv('sorted_wpt_parameters_from_valid_samples_20240924_7.csv', index=False)

print("The sorted DataFrame has been saved as 'sorted_final_mutual_inductance.csv'.")
