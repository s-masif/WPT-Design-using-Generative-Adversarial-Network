import pandas as pd
import re

# Load the dataset with incorrect tensor formatting
correct_data = pd.read_csv(r'C:\Users\Andrew\PycharmProjects\pythonProject\Data\mutual_inductance_dataset_updated_9th_sept.csv')

# Function to clean tensor strings and convert to float
def clean_tensor_string(value):
    if isinstance(value, str):
        # Use regex to extract the floating point number from the tensor string
        match = re.search(r'tensor\(([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)', value)
        if match:
            return float(match.group(1))  # Convert extracted value to float
    return value  # Return the value unchanged if it's not a string tensor

# Apply the cleaning function to all columns that might contain tensor strings
for col in correct_data.columns:
    correct_data[col] = correct_data[col].apply(clean_tensor_string)

# Verify the cleaned data
print(correct_data.head())

# Save the cleaned dataset to a new CSV file
correct_data.to_csv('cleaned_mutual_inductance_dataset_9th_sept_correct_geometry.csv', index=False)
