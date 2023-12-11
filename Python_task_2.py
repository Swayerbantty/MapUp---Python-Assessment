#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

def calculate_distance_matrix(file_path):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)
    
    # Create an empty distance matrix with IDs as both index and columns
    unique_ids = df['id_start'].unique()
    distance_matrix = pd.DataFrame(index=unique_ids, columns=unique_ids, dtype=float)
    
    # Initialize the matrix with zeros on the diagonal
    for id in unique_ids:
        distance_matrix.at[id, id] = 0.0
    
    # Iterate through the DataFrame and calculate cumulative distances
    for index, row in df.iterrows():
        source_id = row['id_start']
        destination_id = row['id_end']
        distance = row['distance']
        
        # Update the distance matrix symmetrically
        distance_matrix.at[source_id, destination_id] = distance
        distance_matrix.at[destination_id, source_id] = distance
    
    # Calculate cumulative distances by adding distances along known routes
    for k in unique_ids:
        for i in unique_ids:
            for j in unique_ids:
                if (distance_matrix.at[i, j] > distance_matrix.at[i, k] + distance_matrix.at[k, j]) or pd.isna(distance_matrix.at[i, j]):
                    distance_matrix.at[i, j] = distance_matrix.at[i, k] + distance_matrix.at[k, j]
    
    return distance_matrix

# Replace 'dataset-3.csv' with the actual path to your CSV file
result_distance_matrix = calculate_distance_matrix('dataset-3.csv')

# Print the resulting distance matrix
print(result_distance_matrix)
result_distance_matri


# In[2]:


import pandas as pd

def unroll_distance_matrix(distance_matrix):
    # Extract the unique IDs from the distance matrix
    unique_ids = distance_matrix.index
    
    # Create an empty DataFrame to store the unrolled distances
    unrolled_distances = pd.DataFrame(columns=['id_start', 'id_end', 'distance'])
    
    # Iterate through unique ID pairs to generate combinations
    for id_start in unique_ids:
        for id_end in unique_ids:
            # Exclude same 'id_start' and 'id_end' pairs
            if id_start != id_end:
                distance = distance_matrix.at[id_start, id_end]
                unrolled_distances = unrolled_distances.append({'id_start': id_start, 'id_end': id_end, 'distance': distance}, ignore_index=True)
    
    return unrolled_distances

# Use the result_distance_matrix from Question 1 as input
result_unrolled_distances = unroll_distance_matrix(result_distance_matrix)

# Print the resulting unrolled distance DataFrame
print(result_unrolled_distances)


# In[4]:


import pandas as pd

def find_ids_within_ten_percentage_threshold(distance_df, reference_value):
    # Calculate the average distance for the reference value
    reference_avg_distance = distance_df[distance_df['id_start'] == reference_value]['distance'].mean()
    
    # Calculate the lower and upper thresholds within 10% of the average
    lower_threshold = reference_avg_distance * 0.9
    upper_threshold = reference_avg_distance * 1.1
    
    # Filter the DataFrame to find values within the threshold
    within_threshold_df = distance_df[(distance_df['id_start'] != reference_value) &
                                      (distance_df['distance'] >= lower_threshold) &
                                      (distance_df['distance'] <= upper_threshold)]
    
    # Get unique values from the 'id_start' column that satisfy the threshold
    unique_values_within_threshold = sorted(within_threshold_df['id_start'].unique())
    
    return unique_values_within_threshold

# Replace 'result_unrolled_distance_df' with the actual DataFrame generated in Question 2
result_within_threshold = find_ids_within_ten_percentage_threshold(result_unrolled_distances, reference_value=1)

# Print the sorted list of values within the 10% threshold
print(result_within_threshold)


# In[5]:


import pandas as pd

def calculate_toll_rate(df):
    # Define rate coefficients for each vehicle type
    rate_coefficients = {
        'moto': 0.8,
        'car': 1.2,
        'rv': 1.5,
        'bus': 2.2,
        'truck': 3.6
    }
    
    # Calculate toll rates for each vehicle type and add new columns
    for vehicle_type, rate_coefficient in rate_coefficients.items():
        df[vehicle_type] = df['distance'] * rate_coefficient
    
    return df

# Replace 'result_unrolled_distance_df' with the actual DataFrame generated in Question 2
result_df_with_toll_rates = calculate_toll_rate(result_unrolled_distances)

# Print the resulting DataFrame with toll rates
print(result_df_with_toll_rates)


# In[6]:


import pandas as pd
import datetime

def calculate_time_based_toll_rates(input_df):
    # Create a DataFrame to store the resulting toll rates
    result_df = input_df.copy()
    
    # Define time ranges and discount factors
    time_ranges = [
        ('00:00:00', '10:00:00', 0.8),
        ('10:00:00', '18:00:00', 1.2),
        ('18:00:00', '23:59:59', 0.8)
    ]
    
    # Iterate through the time ranges and apply discount factors
    for start_time_str, end_time_str, discount_factor in time_ranges:
        start_time = datetime.datetime.strptime(start_time_str, '%H:%M:%S').time()
        end_time = datetime.datetime.strptime(end_time_str, '%H:%M:%S').time()
        
        # Create conditions for weekdays and weekends
        weekdays_condition = (result_df['start_time'] >= start_time) & (result_df['end_time'] <= end_time) & (result_df['start_day'].isin(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']))
        weekends_condition = (result_df['start_time'] >= start_time) & (result_df['end_time'] <= end_time) & (result_df['start_day'].isin(['Saturday', 'Sunday']))
        
        # Apply discount factors based on conditions
        result_df.loc[weekdays_condition, 'vehicle'] = result_df.loc[weekdays_condition, 'vehicle'] * discount_factor
        result_df.loc[weekends_condition, 'vehicle'] = result_df.loc[weekends_condition, 'vehicle'] * 0.7
    
    return result_df

# Example usage:
# Assuming you have the DataFrame result_unrolled_distances from Question 4
# Add start_day, start_time, end_day, and end_time columns to the DataFrame
result_unrolled_distances['start_day'] = 'Monday'
result_unrolled_distances['end_day'] = 'Sunday'
result_unrolled_distances['start_time'] = datetime.datetime.strptime('00:00:00', '%H:%M:%S').time()
result_unrolled_distances['end_time'] = datetime.datetime.strptime('23:59:59', '%H:%M:%S').time()

# Calculate time-based toll rates
result_with_toll_rates = calculate_time_based_toll_rates(result_unrolled_distances)

# Print the resulting DataFrame
print(result_with_toll_rates)


# In[8]:


import pandas as pd
import datetime

def calculate_time_based_toll_rates(input_df):
    # Define the time ranges for weekdays and weekends
    weekday_ranges = [
        (datetime.time(0, 0, 0), datetime.time(10, 0, 0), 0.8),
        (datetime.time(10, 0, 0), datetime.time(18, 0, 0), 1.2),
        (datetime.time(18, 0, 0), datetime.time(23, 59, 59), 0.8)
    ]
    weekend_range = (datetime.time(0, 0, 0), datetime.time(23, 59, 59), 0.7)
    
    # Create a list to store the generated rows
    new_rows = []
    
    # Iterate through unique (id_start, id_end) pairs
    unique_pairs = input_df[['id_start', 'id_end']].drop_duplicates()
    
    for index, row in unique_pairs.iterrows():
        id_start = row['id_start']
        id_end = row['id_end']
        
        # Generate rows for each time range
        for (start_time, end_time, discount_factor) in weekday_ranges:
            new_row = {
                'id_start': id_start,
                'id_end': id_end,
                'start_day': 'Monday',
                'end_day': 'Monday',
                'start_time': start_time,
                'end_time': end_time,
                'discount_factor': discount_factor
            }
            new_rows.append(new_row)
        
        for (start_time, end_time, discount_factor) in weekday_ranges:
            new_row = {
                'id_start': id_start,
                'id_end': id_end,
                'start_day': 'Saturday',
                'end_day': 'Sunday',
                'start_time': start_time,
                'end_time': end_time,
                'discount_factor': discount_factor
            }
            new_rows.append(new_row)
        
        start_time, end_time, discount_factor = weekend_range
        new_row = {
            'id_start': id_start,
            'id_end': id_end,
            'start_day': 'Saturday',
            'end_day': 'Sunday',
            'start_time': start_time,
            'end_time': end_time,
            'discount_factor': discount_factor
        }
        new_rows.append(new_row)
    
    # Create a DataFrame from the generated rows
    result_df = pd.DataFrame(new_rows)
    
    # Merge the input_df with the result_df on (id_start, id_end)
    merged_df = pd.merge(input_df, result_df, on=['id_start', 'id_end'], how='inner')
    
    return merged_df

# Example usage:
# Replace 'input_df' with the DataFrame from Question 4
result_df = calculate_time_based_toll_rates(result_unrolled_distances)

# Print the resulting DataFrame
print(result_df)


# In[14]:


import pandas as pd
import numpy as np
from datetime import datetime, time, timedelta

def calculate_time_based_toll_rates(input_df):
    # Create a DataFrame to store the results
    result_df = pd.DataFrame()

    # Define the time ranges and discount factors
    time_ranges = [
        (time(0, 0, 0), time(10, 0, 0), 0.8),    # Weekdays 00:00 - 10:00
        (time(10, 0, 0), time(18, 0, 0), 1.2),  # Weekdays 10:00 - 18:00
        (time(18, 0, 0), time(23, 59, 59), 0.8) # Weekdays 18:00 - 23:59:59
    ]
    
    weekend_discount_factor = 0.7
    
    # Iterate over each unique (id_start, id_end) pair
    unique_pairs = input_df[['id_start', 'id_end']].drop_duplicates()
    
    for _, row in unique_pairs.iterrows():
        id_start, id_end = row['id_start'], row['id_end']
        
        # Iterate over each day of the week
        for day in range(7):
            start_time = time(0, 0, 0)
            end_time = time(23, 59, 59)
            
            # Apply the discount factor based on the day of the week
            if day < 5:  # Weekdays
                discount_factor = 0.8 if start_time <= time(10, 0, 0) else 1.2
            else:  # Weekends
                discount_factor = weekend_discount_factor
            
            # Create a DataFrame for the current time range
            current_df = input_df[(input_df['id_start'] == id_start) & (input_df['id_end'] == id_end)]
            current_df['start_day'] = datetime.now().replace(day=day+1).strftime('%A')  # Proper case day name
            current_df['end_day'] = datetime.now().replace(day=day+1).strftime('%A')    # Proper case day name
            current_df['start_time'] = start_time
            current_df['end_time'] = end_time
            
            # Apply the discount factor to vehicle columns
            for start, end, factor in time_ranges:
                current_df.loc[(current_df['hour'] >= start.hour) & (current_df['hour'] < end.hour), 'vehicle'] *= factor
            
            # Append the current DataFrame to the result DataFrame
            result_df = pd.concat([result_df, current_df], ignore_index=True)
    
    return result_df

result_df = calculate_time_based_toll_rates(result_unrolled_distances)

# Print the resulting DataFrame
print(result_df)


# Usage example:
# result_df = calculate_time_based_toll_rates(input_df)


# In[ ]:




