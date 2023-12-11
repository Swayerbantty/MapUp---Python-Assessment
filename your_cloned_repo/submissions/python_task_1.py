#!/usr/bin/env python
# coding: utf-8

# In[14]:


import pandas as pd

def generate_car_matrix(file_path):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)
    print(df)
    
    # Pivot the DataFrame to create a matrix
    car_matrix = df.pivot(index='id_1', columns='id_2', values='car')
    
    # Fill NaN values with 0 and set the diagonal values to 0
    car_matrix = car_matrix.fillna(0)
    
    return car_matrix

# Replace 'dataset-1.csv' with the actual file path
result_df = generate_car_matrix('dataset-1.csv')

# Print the result DataFrame
print(result_df)


# In[21]:


import pandas as pd

def get_type_count(df):
    # Create a new categorical column 'car_type' based on the 'car' column values
    df['car_type'] = pd.cut(df['car'], bins=[-float('inf'), 15, 25, float('inf')],
                            labels=['low', 'medium', 'high'], right=False)
    
    # Calculate the count of occurrences for each car_type category
    type_counts = df['car_type'].value_counts().to_dict()
    
    
    # Sort the dictionary alphabetically based on keys
    sorted_type_counts = dict(sorted(type_counts.items()))

    
    return sorted_type_counts

# Read the 'dataset-1.csv' file into a DataFrame
file_path = 'dataset-1.csv'  # Replace with the actual file path
df = pd.read_csv(file_path)

# Call the function and get the count of car_type occurrences
result_dict = get_type_count(df)
print(result_dict)
#print(df)


# In[23]:


import pandas as pd

def get_bus_indexes(df):
    # Calculate the mean value of the 'bus' column
    bus_mean = df['bus'].mean()
    print(bus_mean)
    
    # Find the indices where 'bus' values are greater than twice the mean
    bus_indexes = df[df['bus'] > 2 * bus_mean].index.tolist()
    
    # Sort the indices in ascending order
    bus_indexes.sort()
    
    return bus_indexes

# Read the 'dataset-1.csv' file into a DataFrame
df = pd.read_csv('dataset-1.csv')

# Call the function to get the indices
result = get_bus_indexes(df)

# Print the sorted list of indices
print(result)


# In[26]:


import pandas as pd

def filter_routes(df):
    # Calculate the average of 'truck' values for each unique 'route'
    route_avg_truck = df.groupby('route')['truck'].mean()
    
    # Filter the routes where the average 'truck' value is greater than 7
    filtered_routes = route_avg_truck[route_avg_truck > 7].index.tolist()
    
    # Sort the list of filtered routes in ascending order
    filtered_routes.sort()
    
    
    return filtered_routes

# Read the 'dataset-1.csv' file into a DataFrame
df = pd.read_csv('dataset-1.csv')

# Call the function to get the sorted list of filtered routes
result = filter_routes(df)

# Print the sorted list of routes
print(result)


# In[27]:


import pandas as pd

def multiply_matrix(input_df):
    # Create a deep copy of the input DataFrame to avoid modifying the original
    modified_df = input_df.copy()
    
    # Apply the modification logic
    modified_df[modified_df > 20] *= 0.75
    modified_df[modified_df <= 20] *= 1.25
    
    # Round the values to 1 decimal place
    modified_df = modified_df.round(1)
    
    return modified_df

# Replace 'result_df' with the actual DataFrame from Question 1
# For example, if you obtained the DataFrame using the previous code, use:
# result_df = generate_car_matrix('dataset-1.csv')

# Call the function to get the modified DataFrame
modified_result_df = multiply_matrix(result_df)

# Print the modified DataFrame
print(modified_result_df)


# In[14]:


import pandas as pd

def verify_time_completeness(df):
    # Convert the timestamp columns to datetime objects
    df['start_datetime'] = pd.to_datetime(df['startDay'] + ' ' + df['startTime'],errors = 'coerce')
    df['end_datetime'] = pd.to_datetime(df['endDay'] + ' ' + df['endTime'], errors = 'coerce')
    
    # Create a DataFrame with all possible timestamps for a 7-day period
    all_timestamps = pd.date_range(start='12:00:00 AM', periods=24*60, freq='T')
    all_days_of_week = pd.date_range(start='Monday', periods=7, freq='D').dayofweek
    
    # Create an empty DataFrame to store verification results
    verification_results = pd.DataFrame(columns=['id', 'id_2', 'valid'])
    
    # Iterate through unique (id, id_2) pairs
    for (id, id_2), group in df.groupby(['id', 'id_2']):
        start_times = group['start_datetime']
        end_times = group['end_datetime']
        
        # Check if the timestamps span a full 24-hour period
        full_day_coverage = (start_times.min() <= all_timestamps.min()) and (end_times.max() >= all_timestamps.max())
        
        # Check if the timestamps span all 7 days of the week
        days_of_week_coverage = set(start_times.dt.dayofweek.unique()) == set(all_days_of_week)
        
        # Combine the results to determine if the pair has incorrect timestamps
        valid_pair = full_day_coverage and days_of_week_coverage
        
        # Append the result to the verification DataFrame
        verification_results = verification_results.append({'id': id, 'id_2': id_2, 'valid': valid_pair}, ignore_index=True)
    
    # Set (id, id_2) as a multi-index
    verification_results.set_index(['id', 'id_2'], inplace=True)
    
    return verification_results['valid']

# Read the 'dataset-2.csv' file into a DataFrame
df = pd.read_csv('dataset-2.csv')

# Call the function to get the boolean series
verification_result = verify_time_completeness(df)

# Print the boolean series
print(verification_result)


# In[ ]:




