import pandas as pd

# Load your dataset (update 'your_file_path.xlsx' with the actual file path)
file_path = 'C:/Users/ASUS_ROG/Desktop/thesised/Datasets/Initial/2023/CleanedSpring2023.xlsx'
data = pd.read_excel(file_path)

# Rename columns for clarity and usability
data.columns = ["Date", "Time", "Parameter", "Value", "Unit"]

# Pivot the dataset to arrange parameters as separate columns
formatted_data = data.pivot_table(index=["Date", "Time"], columns="Parameter", values="Value", aggfunc="first").reset_index()

# Rename columns to match the desired order
formatted_data.columns.name = None  # Remove the hierarchical name
formatted_data = formatted_data.rename(columns={
    "pH": "pH",
    "Specific conductance": "Conductivity",
    "Temperature, water": "Water Temperature",
    "Turbidity": "Turbidity",
    "Barometric pressure" : "Barometric Pressure",
    "Total dissolved solids": "TDS",
    "Dissolved Oxygen": "Dissolved Oxygen"
})

# Arrange columns in the desired order
desired_columns = ["Date", "Time", "pH", "Conductivity", "Water Temperature", "Turbidity", "Barometric Pressure", "TDS", "Dissolved Oxygen"]
formatted_data = formatted_data.reindex(columns=desired_columns)

# Export the formatted data to a new Excel file
export_path = 'C:/Users/ASUS_ROG/Desktop/thesised/Datasets/Arranged/SPR23ALL_CL.xlsx'  # Update this with your preferred file path
formatted_data.to_excel(export_path, index=False)

print(f"Formatted dataset exported to {export_path}")

