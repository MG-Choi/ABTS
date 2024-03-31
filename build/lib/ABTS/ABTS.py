#!/usr/bin/env python
# coding: utf-8

# # Library

# In[1]:


import pandas as pd
import numpy as np
import os
import re
import warnings
# from tqdm import tqdmㅇ
from tqdm.auto import tqdm
import random
from shapely.geometry import Point
from pyproj import Geod
import gzip

import geopandas as gpd
import networkx as nx
import osmnx as ox
from geopy.distance import geodesic

# 경고 메시지 무시 설정
warnings.filterwarnings('ignore')


# # ABTS Simulation code

# ## 1. Trip Occurence Builder
# ### 1.0. Data staging (1): The probability of a person with age ‘a’ having ‘t’ trips occurs in a single day

# In[2]:


def naive_bayes_prob_with_day(df, age_col, tripPurpose_col, travday_col):
    """
    Calculate the probability of trip purpose by age group and day type (Weekday or Weekend) using Naive Bayes.

    Parameters:
    - df: DataFrame containing the NHTS data
    - age_col: The name of the column representing age groups
    - tripPurpose_col: The name of the column representing the trip purpose
    - travday_col: The name of the column representing the day type (Weekday or Weekend)

    Returns:
    - result_df: A DataFrame with the calculated probabilities for each combination of age group, trip purpose, and day type
    """

    # Create a DataFrame to store the results
    result_df = pd.DataFrame(columns=['Age', 'Trip_pur', 'Day_Type', 'Prob'])

    # Extract unique values of each Age group, Trip type, and define day types
    unique_ages = df[age_col].unique()
    unique_trips = df[tripPurpose_col].unique()
    day_types = ['Weekday', 'Weekend']
    
    # Total Population
    total_pop = len(df)

    # Loop over each Age group
    for age in unique_ages:
        age_group_pop = len(df[df[age_col] == age])
        
        # Calculate P(X_a / X) - Probability of being in age group 'a' in the total population
        p_xa_x = age_group_pop / total_pop
        
        # Loop over each Trip type
        for trip in unique_trips:
            
            # Also, loop over each Day Type (Weekday, Weekend)
            for day_type in day_types:
                
                # Subset DataFrame based on day type
                if day_type == 'Weekday':
                    sub_df = df[df[travday_col] == 'Weekday']
                else:  # For Weekend
                    sub_df = df[df[travday_col] == 'Weekend']
                
                # Calculate P(θ_t / θ) - Probability of trip type 't' given the day type
                total_trips = sub_df[tripPurpose_col].value_counts().sum()
                p_theta_t_theta = sub_df[sub_df[tripPurpose_col] == trip].shape[0] / total_trips
                
                # Calculate P(X_a | θ_t) - Probability of being in age group 'a' given the trip type 't'
                p_xa_thetat = len(sub_df[(sub_df[age_col] == age) & (sub_df[tripPurpose_col] == trip)]) / sub_df[sub_df[tripPurpose_col] == trip].shape[0]
                
                # Calculate Naive Bayes probability
                prob = (p_xa_thetat * p_theta_t_theta) / p_xa_x
                
                # Setting the result column name based on the day type
                col_name = 'WD_prob' if day_type == 'Weekday' else 'WK_prob'
                
                # Add the result to the DataFrame using concat
                temp_df = pd.DataFrame([{'Age': age, 'Trip_pur': trip, 'Day_Type': day_type, 'Prob': prob}])
                result_df = pd.concat([result_df, temp_df], ignore_index=True)
                
    return result_df


# ### 1.0. Data staging (2): The number of trips based on unique IDs, age, day type, and trip purpose derived from NHTS data

# In[3]:


def generate_trip_distribution(trippub_total, id_age, id_col, day_type_col, trip_purpose_col):
    """
    Generate a distribution of trips based on unique IDs, age, day type, and trip purpose.

    Parameters:
    - trippub_total: DataFrame containing trip data (NHTS)
    - id_age: The name of the column representing the age identifier
    - id_col: The name of the column representing unique identifiers for individuals or entities
    - day_type_col: The name of the column representing the type of the day (e.g., Weekday or Weekend)
    - trip_purpose_col: The name of the column representing the purpose of the trip

    Returns:
    - count_series: A DataFrame that shows the count of trips for each combination of ID, age, day type, and trip purpose.
    """

    # Copy the input DataFrame to avoid modifying the original data
    trip_total = trippub_total.copy()
    
    # Remove rows where the sum of DWELTIME for each combination of PERSONID_new and TRAVDAY_new is 0
    sum_dweltime = trip_total.groupby(['uniqID', 'Day_Type'])['Dwell_T_min'].sum().reset_index()
    valid_ids = sum_dweltime[sum_dweltime['Dwell_T_min'] != 0][['uniqID', 'Day_Type']]
    trip_total = pd.merge(trip_total, valid_ids, on=['uniqID', 'Day_Type'])

    # Aggregate the count of rows
    count_series = trip_total.groupby([id_col, id_age, day_type_col, trip_purpose_col]).size().reset_index(name='count')
    
    # The following line is commented out as it seems redundant given that 'day_type_col' already specifies day type
    # count_series['TRAVDAY_new'] = count_series[day_type_col].apply(lambda x: 'Weekday' if x == 'Weekday' else 'Weekend')
    
    return count_series


# ### 1.1. The number of trips occurring ‘k’ times for a single individual ‘i’ in a day

# In[4]:


def generate_combined_trip_count(naive_prob, trip_count, age_n_dict, method, Home_cbg, W_k_weekday=1.0, W_k_weekend=1.0,
                                 W_t_weekday=None, W_t_weekend=None, print_progress=True):
    """
    Generate a combined trip count distribution based on age groups, day type, and trip purpose.
    
    Parameters:
    - naive_prob: DataFrame containing Naive Bayes probabilities.
    - trip_count: DataFrame with trip count data.
    - age_n_dict: Dictionary mapping age groups to the number of individuals.
    - method: String specifying the method to distribute trips ('multinomial' or 'cdf').
    - Home_cbg: The home census block group id.
    - W_k_weekday: Weight multiplier for trip counts on weekdays.
    - W_k_weekend: Weight multiplier for trip counts on weekends.
    - W_t_weekday: Dictionary with trip purpose as keys and weights as values for weekdays.
    - W_t_weekend: Dictionary with trip purpose as keys and weights as values for weekends.
    - print_progress: Boolean flag to print progress.
    
    Returns:
    - combined_result_df: DataFrame with the generated trip distribution.
    """
    
    # Print initial message if progress printing is enabled
    if print_progress:
        print('<Trip occurrence builder>')
    
    # Initialize weight dictionaries if not provided
    if W_t_weekday is None:
        W_t_weekday = {}  # Default weekday weights as empty dict
    if W_t_weekend is None:
        W_t_weekend = {}  # Default weekend weights as empty dict
        
    
    # Create an empty DataFrame to store results
    combined_result_df = pd.DataFrame()

    # Define extended day types to include specific days of the week
    extended_day_types = {
        'Weekday': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'],
        'Weekend': ['Saturday', 'Sunday']
    }
    
    
    # Define a nested function to generate trip counts for each age group and day type
    def generate_trip_count(naive_result_total, trip_count, age_group, n, method, extended_day_types, start_id=0):
        result_list = []  # Initialize an empty list to store intermediate results

        # Iterate over each base day type and its actual days
        for base_day_type, actual_days in extended_day_types.items():
            # Filter trip counts for the current age group and base day type
            trip_count_df = trip_count[(trip_count['age_class'] == age_group) & (trip_count['Day_Type'] == base_day_type)]
            # Aggregate trip counts by person
            trip_count_by_person = trip_count_df.groupby(['uniqID']).agg({'count': 'sum'}).reset_index()


            # Filter Naive Bayes probabilities for the current age group and base day type
            prob_df = naive_result_total[(naive_result_total['Age'] == age_group) & (naive_result_total['Day_Type'] == base_day_type)]

            # Iterate over each actual day
            for actual_day in actual_days:
                # Repeat process 'n' times for simulation
                for i in range(n):
                    current_id = start_id + i  # Unique identifier for each simulation iteration
                    
                    # Determine trip count multiplier based on day type
                    if actual_day in ['Sunday', 'Saturday']:  # Weekend
                        theta_i = np.random.choice(trip_count_by_person['count']) * W_k_weekend
                    else:  # Weekday
                        theta_i = np.random.choice(trip_count_by_person['count']) * W_k_weekday
                    
                    # Adjust probabilities with weights if they exist for the trip purpose
                    adjusted_prob_df = prob_df.copy()
                    
                    if actual_day == 'Sunday' or actual_day == 'Saturday': # weekend
                        for trip_purpose, weight in W_t_weekday.items():
                            adjusted_prob_df.loc[adjusted_prob_df['Trip_pur'] == trip_purpose, 'Prob'] *= weight
                    else:
                        for trip_purpose, weight in W_t_weekend.items():
                            adjusted_prob_df.loc[adjusted_prob_df['Trip_pur'] == trip_purpose, 'Prob'] *= weight

                    # Normalize probabilities again after weighting
                    adjusted_prob_df['Prob'] /= adjusted_prob_df['Prob'].sum()

                    if method == 'multinomial':
                        trips = np.random.multinomial(theta_i, adjusted_prob_df['Prob'])
                    elif method == 'cdf':
                        cdf = np.cumsum(adjusted_prob_df['Prob'])
                        trips = np.histogram(np.random.rand(theta_i), bins=[0] + list(cdf), range=(0,1))[0]
                    
                    
                    home_count = trips[prob_df['Trip_pur'] == 'Home'][0]
                    
                    # Append results for each trip purpose
                    for t, count in zip(prob_df['Trip_pur'], trips):
                        result_list.append({'uniqID': current_id, 'ageGroup': age_group, 'Day_Type': actual_day, 'Week_Type': base_day_type, 'TRPPURP': t, 'count': count})

        # Convert the list of results into a DataFrame
        result_df = pd.DataFrame(result_list)
        return result_df

    # Initialize the starting unique ID for simulation
    current_max_id = 0

    # Decide whether to use tqdm for progress indication based on the print_progress flag
    iterator = tqdm(age_n_dict.items(), desc='1. Processing age groups to generate trip counts') if print_progress else age_n_dict.items()

    # Iterate over each age group and its associated number of individuals
    for age_group, n in iterator:
        temp_df = generate_trip_count(naive_prob, trip_count, age_group, n, method, extended_day_types, current_max_id)
        combined_result_df = pd.concat([combined_result_df, temp_df], ignore_index=True)
        current_max_id += n  # Update the current_max_id for the next batch

    if print_progress:
        tqdm.pandas(desc="2. Adjust home counts")
    
    # Function to adjust home count in each group
    def adjust_home_count(group):
        total_count = group['count'].sum()
        home_count_row = group[group['TRPPURP'] == 'Home']

        # Ensure that there are at least 2 home counts if total count is 2 or more
        if total_count >= 2 and home_count_row['count'].iloc[0] < 2:
            group.loc[group['TRPPURP'] == 'Home', 'count'] = 2
        return group

    # Apply the function to adjust home counts
    if print_progress:
        combined_result_df = combined_result_df.groupby(['uniqID', 'Day_Type']).progress_apply(adjust_home_count).reset_index(drop=True)
    else:
        combined_result_df = combined_result_df.groupby(['uniqID', 'Day_Type']).apply(adjust_home_count).reset_index(drop=True)

    # Assign the Home_cbg to all rows
    combined_result_df["Home_cbg"] = Home_cbg
    
    # columns: Wt amd Wk
    
    
    # Function to retrieve weight for a given trip purpose from the weight dictionaries
    def get_weight(trppurp, W_t):
        return W_t.get(trppurp, 1.0)  # Return 1.0 if the trip purpose is not in the dictionary

    # Assign weekday and weekend trip count weights to all rows
    combined_result_df['Wk_wD'] = W_k_weekday
    combined_result_df['Wk_wK'] = W_k_weekend

    # Apply the get_weight function to assign trip purpose weights for weekdays and weekends
    combined_result_df['Wt_wD'] = combined_result_df['TRPPURP'].apply(lambda x: get_weight(x, W_t_weekday))
    combined_result_df['Wt_wK'] = combined_result_df['TRPPURP'].apply(lambda x: get_weight(x, W_t_weekend))

    return combined_result_df


# ## 2. Trip Chains Builder
# ### 2.0. Data staging (1): Create trip seqeunce of individuals using origin NHTS data

# In[5]:


def create_trip_sequence(df, print_progress=True):
    """
    Creates a trip sequence column by concatenating Trip_pur values for each group defined by age_class, Day_Type, and uniqID.
    
    Parameters:
    - df: DataFrame containing trip data (NHTS).
    - print_progress: Boolean indicating whether to print progress information.

    Returns:
    - DataFrame with an added column 'Trip_sequence' representing the sequence of trip purposes.
    """

    # Function to aggregate counts after trip sequence generation
    def aggregate_count(df):
        # Group by relevant columns and aggregate 'count' values
        aggregated_df = df.groupby(['age_class', 'Day_Type', 'uniqID', 'Trip_sequence']).agg({'count': 'sum'}).reset_index()
        return aggregated_df
    
    # Sort the DataFrame by start time to ensure trip sequence is chronological
    sorted_df = df.sort_values('sta_T_hms')
    
    # Group the sorted DataFrame by age class, day type, and unique ID
    grouped = sorted_df.groupby(['age_class', 'Day_Type', 'uniqID'])
    
    # Initialize a list to store result data
    result_data = []
    
#     print("create trip sequence...")

    # Check if progress should be printed, wrap the iterator with tqdm if true
    iterator = tqdm(grouped, desc='Creating trip sequences derived from data...') if print_progress else grouped

    # Iterate through each group
    for name, group in iterator:
        # Concatenate the 'Trip_pur' values to form the trip sequence
        trip_sequence = '-'.join(group['Trip_pur'])
        # Append the result as a dictionary to the result_data list
        result_data.append({
            'age_class': name[0],
            'Day_Type': name[1],
            'uniqID': name[2],
            'Trip_sequence': trip_sequence,
            # Set initial count to 1 for each unique sequence
            'count': 1
        })
    
    # Convert the result_data list to a DataFrame
    result_df = pd.DataFrame(result_data)
    
    # Aggregate counts in case there are duplicate sequences
    result_df = aggregate_count(result_df)
    
    # Return the final DataFrame with aggregated trip sequences
    return result_df


# ### 2.1) Finding optimal origin sequence O_i most similar to S_i
# ### 2.2) Randomly assign the trip sequence for S_i
# ### 2.3) Reassign the sequence of trip in S_i based on O_i

# In[6]:


def makeTripSequence(trip_sequence_simul, trip_sequence_origin, print_progress=True):
    """
    Constructs and refines trip sequences for simulated data to ensure realistic patterns by aligning them with sequences from original NHTS data. 
    The process includes adjusting 'Home' trip counts to reflect real-world patterns, duplicating rows based on trip counts, 
    finding and assigning the most similar trip sequence from NHTS data, and sequentially refining trip sequences to eliminate inconsistencies 
    such as duplicated or consecutive 'Home' trips and ensuring all trip sequences start and end with 'Home'. 
    The function applies several passes of adjustments to achieve coherent and logically ordered trip sequences that realistically simulate daily travel behavior.

    Parameters:
    - trip_sequence_simul: DataFrame containing simulated trip data that needs sequence construction and adjustment.
    - trip_sequence_origin: DataFrame with original trip sequences from NHTS data, used as a reference for similarity comparison.
    - print_progress: Boolean flag indicating whether to print progress messages and bars during the execution.

    Returns:
    - A DataFrame with refined trip sequences for each individual and day, ready for further analysis or simulation tasks.
    """
    
    
    # Print initial message if progress printing is enabled
    if print_progress:
        print('<Trip chain builder>')
    
    # Adjust home count in the simulated trip sequences to match real trip patterns
    tqdm.pandas(desc="1. Split trips purpose to each row")
    def adjust_home_count(group):
        # Calculate the sum of non-home trip counts and adjust home count accordingly
        non_home_count_sum = group.loc[group['TRPPURP'] != 'Home', 'count'].sum()
        new_home_count = non_home_count_sum + 1

        # Adjust the home count in the DataFrame
        home_idx = group.loc[group['TRPPURP'] == 'Home'].index
        if len(home_idx) > 0:  # If there is a home trip
            home_idx = home_idx[0]
            if group.at[home_idx, 'count'] > new_home_count:
                group.at[home_idx, 'count'] = new_home_count
            group = group[group['count'] > 0]

        # Duplicate rows based on the 'count' value to create individual trip records
        new_rows = []
        for _, row in group.iterrows():
            new_rows.extend([row] * int(row['count']))  # Duplicate row 'count' times
        new_group = pd.DataFrame(new_rows).reset_index(drop=True)
        new_group['count'] = 1  # Reset count to 1 for all rows

        return new_group

    # Apply adjust_home_count function to each group
    if print_progress == True:
        adjusted_simul_df = trip_sequence_simul.groupby(['Day_Type', 'uniqID']).progress_apply(adjust_home_count).reset_index(drop=True)
    else:
        adjusted_simul_df = trip_sequence_simul.groupby(['Day_Type', 'uniqID']).apply(adjust_home_count).reset_index(drop=True)
#     --------------------#
    
    tqdm.pandas(desc="2. Find similar sequence from NHTS data") 
    
    def find_most_similar_sequence(query_seq, available_seqs): #1. find trip sequence in origin NHTS, similar to the simulated trips
        
        max_similarity = 0
        most_similar = None

        for seq in available_seqs:
            
            similarity = sum(x == y for x, y in zip(query_seq, seq)) # Calculate string similarity
            if similarity > max_similarity:
                max_similarity = similarity
                most_similar = seq

        return most_similar

    
    unique_trip_seqs = trip_sequence_origin['Trip_sequence'].unique()
    
    def get_sequence(group):
        # Assign the most similar trip sequence from NHTS data to each group
        if len(group) == 1:
            group['seq_similar_orig'] = 1
            return group

        query_seq = "-".join(group['TRPPURP'].tolist())
#         print(query_seq)
        
        # 가장 비슷한 sequence 찾기
        most_similar_seq = find_most_similar_sequence(query_seq, unique_trip_seqs)
        group['seq_similar_orig'] = most_similar_seq
        return group

    # Apply get_sequence function to each group
    if print_progress == True:
        df_with_seq = adjusted_simul_df.groupby(['uniqID', 'ageGroup', 'Day_Type']).progress_apply(get_sequence).reset_index(drop=True)
    else:
        df_with_seq = adjusted_simul_df.groupby(['uniqID', 'ageGroup', 'Day_Type']).apply(get_sequence).reset_index(drop=True)
    

    def MakeSequence(df): 
        # This function constructs trip sequences for individuals based on their simulated trip purposes and the most similar trip sequence from the original data.

        # Initialize an empty column for the sequence
        df['sequence'] = None

        # Define an iterator for grouping by unique IDs and Day_Type, to process each individual's trips per day
        iterator = tqdm(df.groupby(['uniqID', 'Day_Type']), desc='3. Assign trip sequence to individuals') if print_progress else df.groupby(['uniqID', 'Day_Type'])
    
        for name, group in iterator:
        # If the group size is 1, it implies there's only a single trip purpose, simplifying the sequence assignment
            if  group['seq_similar_orig'].iloc[0] == 1:
#                 group['sequence'].iloc[0] = 1
                
                df.loc[group.index, 'sequence'] = 1
            else:
                seq_similar_orig = group['seq_similar_orig'].iloc[0].split('-')
                n = len(group)
                sequence = []

                # Count 'Home' trips and assign them to the first and last sequence positions
                home_count = group['TRPPURP'].value_counts().get('Home', 0) 
                home_assigned = 0
                for trppurp in group['TRPPURP']:
                    if trppurp == 'Home':
                        home_assigned += 1
                        sequence.append(1 if home_assigned == 1 else n)   # Ensure 'Home' appears first and last in the sequence.

                # Process remaining trip purposes not assigned as 'Home'
                remaining = [trppurp for trppurp in group['TRPPURP'] if trppurp != 'Home']
#                 print(seq_similar_orig)
                remaining_indices = sorted(set(range(2, n)) - set(sequence))

#                 display(group)             
                
                for trppurp in remaining: # Initially assign a random sequence to remaining trip purposes
                    random_seq = random.choice(remaining_indices)
                    remaining_indices.remove(random_seq)
                    sequence.append(random_seq)

                    
                for idx in range(1, len(seq_similar_orig)): # Reassign sequence numbers based on the original sequence to maintain trip order
                    prev_purpose, current_purpose = seq_similar_orig[idx-1], seq_similar_orig[idx]

                    # prev_purpose의 sequence를 찾고, 그 다음 sequence를 current_purpose에 할당
                    for seq_num, purp in sorted(list(enumerate(sequence)), key=lambda x: x[1]):
                        if purp == prev_purpose:
                            next_seq = seq_num + 1
                            if next_seq in sequence:
                                continue  # Skip if already assigned

                            # Find and set the next sequence number for current trip purpose based on its predecessor
                            for cur_seq_num, cur_purp in sorted(list(enumerate(sequence)), key=lambda x: x[1]):
                                if cur_purp == current_purpose:
                                    sequence[cur_seq_num] = next_seq
                                    break

                            break  # Stop after reassigning the current purpose   
                
                # Update the DataFrame with the newly assigned sequences
                df.loc[group.index, 'sequence'] = sequence

        return df
    
    df_with_seq = MakeSequence(df_with_seq)
    

    #-------------------------
    
    def print_4():
        for i in tqdm(range(1), desc = '4. Organize and arrange tables'):
        # This loop does nothing but show progress. It's a visual indicator and does not perform any data manipulation.
            None
    
    if print_progress == True:
        print_4()
    
    #---------------
    
    def removeDuplHome(df): 
        # This function removes duplicated 'Home' trip purposes that occur consecutively in the trip sequence, except for the first and last instance.
        # It ensures that each trip sequence correctly reflects a realistic pattern of leaving from and returning to home at most once during the trip sequence.

        
        # Iterate through each group based on 'uniqID' and 'Day_Type'
        iterator = tqdm(df.groupby(['uniqID', 'Day_Type']), desc = ' - 4.1. remove duplicated Home trip') if print_progress else df.groupby(['uniqID', 'Day_Type'])
        
        for name, group in iterator:

            if len(group) == 1:
                continue
            # Calculate the number of 'Home' occurrences in seq_similar_orig
            seq_similar_orig_count = group['seq_similar_orig'].iloc[0].split('-').count('Home') - 2  # Exclude first and last
            if seq_similar_orig_count < 0:
                seq_similar_orig_count = 0

            # Get the count of 'Home' in the group
            home_count = group['TRPPURP'].value_counts().get('Home', 0)

            # Calculate how many 'Home' entries to remove, excluding the first and last
            home_to_remove = home_count - 2 - seq_similar_orig_count

            # Remove excess 'Home' occurrences
            if home_to_remove > 0:
                index_to_remove = group[group['TRPPURP'] == 'Home'].index[1:-1][:home_to_remove]
                df.drop(index_to_remove, inplace=True)   

        return df
    
    df_removed_dupl_Home = removeDuplHome(df_with_seq)
    
    
    def setHomeSequence(df): 
        # Adjusts the sequence of 'Home' trips for each group of trips by a unique individual on a specific day. 
        # If there are exactly two 'Home' entries, their sequences are set to 1 and the last sequence number, ensuring that trips start and end at 'Home'.
        
        # Iterate through each group based on 'uniqID' and 'Day_Type'
        iterator = tqdm(df.groupby(['uniqID', 'Day_Type']), desc = ' - 4.2. reassign Home trip sequence') if print_progress else df.groupby(['uniqID', 'Day_Type'])
        
        for name, group in iterator:

            # If there are exactly 2 'Home' entries, set their sequence to 1 and the total number of rows in the group
            if group['TRPPURP'].value_counts().get('Home', 0) == 2:
                home_indices = group[group['TRPPURP'] == 'Home'].index.tolist()
                # Set the sequence of the first 'Home' entry to 1
                df.loc[home_indices[0], 'sequence'] = 1
                
                # Set the sequence of the second 'Home' entry to the length of the group, making it the last trip
                df.loc[home_indices[1], 'sequence'] = len(group)

        return df
    
    df_setHome = setHomeSequence(df_removed_dupl_Home)
    
    
    def reassignMiddleHomeSequences(df):
        # For trip sequences with more than two 'Home' entries, this function reassigns the sequences of middle 'Home' entries.
        # The aim is to distribute these 'Home' trips more realistically within the sequence, avoiding consecutive 'Home' trips and ensuring that they are placed appropriately among other trip purposes.

        # Iterate through each group based on 'uniqID' and 'Day_Type'
        iterator = tqdm(df.groupby(['uniqID', 'Day_Type']), desc = ' - 4.3. Reassign Home trips if more than 3 trips') if print_progress else df.groupby(['uniqID', 'Day_Type'])
        
        for name, group in iterator:

            # If there are more than 2 'Home' entries
            if group['TRPPURP'].value_counts().get('Home', 0) > 2:

                # Get the index for all 'Home' entries
                home_indices = group[group['TRPPURP'] == 'Home'].index.tolist()

                # Exclude the first and the last 'Home' entries
                middle_home_indices = home_indices[1:-1]

                # Calculate the range for random sequence values
                last_sequence = group['sequence'].max()
                possible_sequences = list(range(3, last_sequence - 1)) #

                if len(possible_sequences) < len(middle_home_indices):
                    # If there are not enough possible sequence numbers, remove excess rows
                    df.drop(middle_home_indices[len(possible_sequences):], inplace=True)
                    middle_home_indices = middle_home_indices[:len(possible_sequences)]

                random.shuffle(possible_sequences)

                # Assign random sequence values to the middle 'Home' entries
                for i, index in enumerate(middle_home_indices):
                    df.loc[index, 'sequence'] = possible_sequences[i]

        return df

    # Example usage
    df_with_middle_home_reassigned = reassignMiddleHomeSequences(df_setHome)

    def reassignConsecutiveSequences(df, num): 
        # Reassigns sequences to ensure they are consecutive without any duplicates or gaps, especially after prior adjustments might have created irregularities in the sequence numbering.

        # Iterate through each group based on 'uniqID' and 'Day_Type'        
        iterator = tqdm(df.groupby(['uniqID', 'Day_Type']), desc = ' - 4.' + str(num+3) + '. Reassign consecutive numbers of sequence_' + str(num)) if print_progress else df.groupby(['uniqID', 'Day_Type'])
        
        for name, group in iterator:

            # Sort by sequence
            sorted_group = group.sort_values('sequence')
            new_sequence = 1
            prev_sequence = None

            for index, row in sorted_group.iterrows():
                # If current sequence is same as previous, increment for non-'Home' or drop 'Home'
                if row['sequence'] == prev_sequence:
                    if row['TRPPURP'] == 'Home':
                        # If the TRPPURP is 'Home', remove the row
                        df.drop(index, inplace=True)
                    else:
                        # If the TRPPURP is not 'Home', increment the sequence by 1
                        new_sequence += 1

                df.loc[index, 'sequence'] = new_sequence
                prev_sequence = row['sequence']
                new_sequence += 1

        return df

    
    df_with_consecutive_sequences = reassignConsecutiveSequences(df_with_middle_home_reassigned, 1)
    df_with_consecutive_sequences = reassignConsecutiveSequences(df_with_consecutive_sequences, 2)

    # sort and reorganize -----------------

    
    # rename column: 'seq_similar_orig' -> seq_NHTS
    df_with_consecutive_sequences.rename(columns={'seq_similar_orig': 'seq_NHTS'}, inplace=True)

    columns = ['uniqID', 'ageGroup', 'Home_cbg', 'Day_Type', 'Week_Type', 'TRPPURP', 'sequence', 'seq_NHTS']
    
    # ALl columns of df_with_consecutive_sequences DataFrame
    all_columns = df_with_consecutive_sequences.columns.tolist()

    # Extract all the columns except selected column
    remaining_columns = [col for col in all_columns if col not in columns]

    # Create new columns sequence list by adding other columns after selected column
    new_column_order = columns + remaining_columns

    # Now reassign this list of columns to dataframe
    organized_df = df_with_consecutive_sequences.reindex(columns=new_column_order)

    # Eradicate 'count' column
    if 'count' in organized_df.columns:
        organized_df.drop('count', axis=1, inplace=True)    
    
#     organized_df = df_with_consecutive_sequences[columns]
    
      # Sort uniqID, Day_Type, sequence by Ascending order 
    sorted_df = organized_df.sort_values(
        by=['uniqID', 'Day_Type', 'sequence'], 
        ascending=[True, True, True]
    )  
    
    sorted_df.reset_index(drop = True, inplace = True)
    
    
        
    tqdm.pandas(desc=" - 4.6. drop consecutive homes except start and end") 
    
    # If there are duplicate Home trips -> delete second one
    def drop_consecutive_homes(group):
        homes = group['TRPPURP'] == 'Home'
        consecutive = homes & homes.shift(fill_value=False)
        drop_indices = consecutive[consecutive].index
        return group.drop(drop_indices)

    if print_progress == True:
        droped_consecutive_homes_df = sorted_df.groupby(['uniqID', 'Day_Type']).progress_apply(drop_consecutive_homes).reset_index(drop=True)
    else:
        droped_consecutive_homes_df = sorted_df.groupby(['uniqID', 'Day_Type']).apply(drop_consecutive_homes).reset_index(drop=True)

    # Reassign sequence
    
    
    tqdm.pandas(desc=" - 4.7. reassign sequence") 
    def reset_sequence(group):
        group['sequence'] = range(1, len(group) + 1)
        return group

    if print_progress == True:
        reassigned_sequence_df = droped_consecutive_homes_df.groupby(['uniqID', 'Day_Type']).progress_apply(reset_sequence).reset_index(drop=True)
    else:
        reassigned_sequence_df = droped_consecutive_homes_df.groupby(['uniqID', 'Day_Type']).apply(reset_sequence).reset_index(drop=True)
    
    #------------------------------- fix errors
    
    tqdm.pandas(desc=" - 4.8. fix home error if need ... (1)") 
    
    # Add new row if the last TRPPURP is not a 'Home' in each group
    def add_home_row_if_needed(group):
        if group['TRPPURP'].iloc[-1] != 'Home':
            new_row = group.iloc[-1].copy()
            new_row['TRPPURP'] = 'Home'
            new_row['sequence'] = new_row['sequence'] + 1
            group = pd.concat([group, pd.DataFrame([new_row])])
        return group

    if print_progress == True:    
        fix_1_df = reassigned_sequence_df.groupby(['uniqID', 'Day_Type']).progress_apply(add_home_row_if_needed).reset_index(drop=True)
    else:
        fix_1_df = reassigned_sequence_df.groupby(['uniqID', 'Day_Type']).apply(add_home_row_if_needed).reset_index(drop=True)
    
    
    tqdm.pandas(desc=" - 4.9. fix home error if need ... (2)") 
    
    def add_home_to_start_if_needed(group):
        if group['TRPPURP'].iloc[0] != 'Home':
            new_row = group.iloc[0].copy()
            new_row['TRPPURP'] = 'Home'
            new_row['sequence'] = 1
            group = pd.concat([new_row.to_frame().T, group])
            group['sequence'] = group['sequence'].astype(int) + 1
            group['sequence'].iloc[0] = 1
        return group

    if print_progress == True:        
        fix_2_df = fix_1_df.groupby(['uniqID', 'Day_Type']).progress_apply(add_home_to_start_if_needed).reset_index(drop=True)
    else:
        fix_2_df = fix_1_df.groupby(['uniqID', 'Day_Type']).apply(add_home_to_start_if_needed).reset_index(drop=True)

    
    return fix_2_df


# ## 3. Trip Timing Estimator
# ### 3.0. Data staging (1): Extracting dwell time by trip purpose using NHTS

# In[7]:


def dwellTime_listFromNHTS(df):
    """
    Extracts and organizes dwell time distributions by trip count, trip purpose, and other classifications from NHTS data. 
    This function is intended to run once to prepare the data for further simulation.

    Parameters:
    - df: DataFrame containing NHTS trip data.

    Returns:
    - A dictionary with dwell time lists categorized by age class, day type, trip count class, and trip purpose.
    """
    # 1. Calculate trip count: Count the number of trips for each unique ID and day type.
    trippub = df.copy()
    grouped = trippub.groupby(['uniqID', 'Day_Type'])
    trippub['tripCount'] = grouped['Trip_pur'].transform('count')  # Add trip count for each row based on grouping.

    # 2. Create trip count class: Classify trip counts into three categories based on the number of trips per day.
    # Categories are defined as 1-3 trips, 4-5 trips, and 6 or more trips, considering trips start and end at home.
    trippub['tripCount_class'] = pd.cut(trippub['tripCount'], bins=[0, 4, 6, float('inf')], labels=[1, 2, 3], right=False)
    # The bins parameter defines the range of trip counts for each class: 
    # - Class 1 for 1-3 trips (inclusive of starting and ending at home)
    # - Class 2 for 4-5 trips
    # - Class 3 for 6 or more trips.
    # These classifications help in understanding the distribution of trip counts and corresponding dwell times.

    dwellTime_dict = trippub.groupby(['age_class', 'Day_Type', 'tripCount_class', 'Trip_pur'])['Dwell_T_min'].apply(list).to_dict()
    
    return dwellTime_dict


# ### 3.0. Data staging (2): Extracting trip start time by trip purpose using NHTS

# In[8]:


def startTime_listFromNHTS(df):
    """
    Generates a dictionary mapping the start times of trips based on age class, day type, and the second trip purpose from the NHTS data.
    This function groups the data and extracts start times to understand typical trip start patterns. 
    It's designed to be run once and reused for analysis or simulation purposes.

    Parameters:
    - df: DataFrame containing NHTS trip data.

    Returns:
    - A dictionary where each key is a tuple of (age class, day type, trip purpose) and each value is a list of 
      non-zero start times (in minutes from midnight) for that combination.
    """
    
    # Initialize an empty dictionary to store start times
    startTime_dict = {}
    trippub_new = df.copy()

    # Group the data by unique ID, age class, and day type. This aggregation is suited for analysis on how 
    # start times may vary across different demographics and types of days (e.g., weekdays vs. weekends).
    grouped_trippub = trippub_new.groupby(['uniqID', 'age_class', 'Day_Type'])

    # Iterate through each group to collect start times
    for (_, age_class, day_type), group in tqdm(grouped_trippub, desc='Make dict of starting time derived from NHTS data'):
        # Only consider groups with more than one trip to ensure we're looking at subsequent trips
        if len(group) > 1:
            # Extract the trip purpose of the second trip in the sequence. This choice focuses on the start time 
            # of the day's first major trip after potentially leaving home.
            trip_pur = group['Trip_pur'].iloc[1]
            key = (age_class, day_type, trip_pur)  # Define a unique key for the dictionary

            # Initialize the list in the dictionary if the key doesn't exist
            if key not in startTime_dict:
                startTime_dict[key] = []

            # Add non-zero start times to the list for this key. Zero values are excluded to avoid considering 
            # trips that might not represent actual departures (e.g., midnight or incorrectly recorded times).
            non_zero_values = [value for value in group['sta_T_min'].tolist() if value != 0]
            startTime_dict[key].extend(non_zero_values)
        
    return startTime_dict


# ### 3.1) Estimating dwell time
# ### 3.2) Estimating trip start time

# In[9]:


def assignDwellStartT(df, dwellTime_dict, startTime_dict, print_progress=True):
    """
    Estimates and assigns dwell times and start times for simulated trip data using distributions derived from NHTS data.
    It first classifies each trip within the simulated data by trip count and then assigns dwell times based on
    age class, day type, trip count class, and trip purpose. Finally, it assigns start times to the trips using
    similar criteria.

    Parameters:
    - df: DataFrame containing the simulated trip data.
    - dwellTime_dict: Dictionary containing dwell time distributions from NHTS data.
    - startTime_dict: Dictionary containing start time distributions from NHTS data.
    - print_progress: Boolean flag to print progress messages.

    Returns:
    - DataFrame with 'Dwell_Time' and 'sta_T_min' (start time in minutes from midnight) assigned for each trip.
    """
    
    if print_progress == True:
        print('<Trip timing estimator>')
    
    # Function to assign dwell time to each trip based on the dwellTime_dict distributions
    def assignDwellTime(df, dwellTime_dict, print_progress):
        simul_trip_sequence = df.copy()

        tqdm.pandas(desc="1. classify tripCount of simulated data")

        # Classify and assign trip count class to each trip based on the size of each group (unique ID and day type)
        group_counts = simul_trip_sequence.groupby(['uniqID', 'Day_Type']).size().to_dict()

        def assign_class(row):
            group_size = group_counts[(row['uniqID'], row['Day_Type'])]

            if group_size <= 3:
                return 1
            elif 4 <= group_size <= 5:
                return 2
            else:
                return 3

        # Assign dwell time to each trip by randomly selecting from the corresponding distribution in dwellTime_dict
        if print_progress == True:
            simul_trip_sequence['trip_count_class'] = simul_trip_sequence.progress_apply(assign_class, axis=1)
        else:
            simul_trip_sequence['trip_count_class'] = simul_trip_sequence.apply(assign_class, axis=1)


        # 2. Sampling Dwell_Time from distribution
        tqdm.pandas(desc="2. Assign Dwell time")

        def assign_dwell_time(row):
            # Convert day type to 'Weekday' or 'Weekend' for consistency with the dictionary keys
            day_type_map = {
                'Monday': 'Weekday',
                'Tuesday': 'Weekday',
                'Wednesday': 'Weekday',
                'Thursday': 'Weekday',
                'Friday': 'Weekday',
                'Saturday': 'Weekend',
                'Sunday': 'Weekend'
            }
            day_type = day_type_map[row['Day_Type']]

            # Construct the key for the dictionary lookup
            key = (row['ageGroup'], day_type, row['trip_count_class'], row['TRPPURP'])
            dwell_times = dwellTime_dict.get(key, [0])

            if not isinstance(dwell_times, (list, np.ndarray)) or len(dwell_times) == 0:
                dwell_time_sample = np.random.randint(10, 301)
                print(f"cannot find from dic, put random dwelltime...({row['ageGroup']}, {row['Day_Type']}, {row['trip_count_class']}, {row['TRPPURP']}, value: {dwell_time_sample})")
                return dwell_time_sample

            dwell_time_sample = -1
            while dwell_time_sample < 0:
                dwell_time_sample = np.random.choice(dwell_times)

            return dwell_time_sample
        
        # Apply function to assign dwell time to each trip
        if print_progress == True:
            simul_trip_sequence['Dwell_Time'] = simul_trip_sequence.progress_apply(assign_dwell_time, axis=1)
        else:
            simul_trip_sequence['Dwell_Time'] = simul_trip_sequence.apply(assign_dwell_time, axis=1)

        # drop column seq_NHTS
        if 'seq_NHTS' in simul_trip_sequence.columns:
            simul_trip_sequence = simul_trip_sequence.drop('seq_NHTS', axis=1)


        return simul_trip_sequence
    
    # Apply the function to assign dwell times
    dwell_table = assignDwellTime(df, dwellTime_dict, print_progress)
    
    # Function to assign start times to each trip
    def assignStartTime(df, startTime_dict, print_progress):

        simul_trip_sequence_dwell_time = df.copy()
        simul_trip_sequence_dwell_time['sta_T_min'] = np.nan

        tqdm.pandas(desc="3. Assign start time")

        # Function to assign start times within each group
        def assign_sta_T_min(group):
            if len(group) >= 2:
                # Set the start time of the first trip to 0 (midnight)
                first_row = group.iloc[0]
                group.at[first_row.name, 'sta_T_min'] = 0

                # Assign start time to the second trip based on startTime_dict
                second_row = group.iloc[1]
                key = (second_row['ageGroup'], second_row['Week_Type'], group['TRPPURP'].iloc[1])
                if key in startTime_dict:
                    group.at[second_row.name, 'sta_T_min'] = np.random.choice(startTime_dict[key])
            else: 
                # For groups with only one trip, assign a start time of 0
                first_row = group.iloc[0]
                group.at[first_row.name, 'sta_T_min'] = 0

            return group


        if print_progress == True:
            # Apply 'assign_sta_T_min' to each group of trips by unique ID and day type
            simul_trip_sequence_2_start_time = simul_trip_sequence_dwell_time.groupby(['uniqID', 'Day_Type']).progress_apply(assign_sta_T_min)
        else:
            simul_trip_sequence_2_start_time = simul_trip_sequence_dwell_time.groupby(['uniqID', 'Day_Type']).apply(assign_sta_T_min)
            
        simul_trip_sequence_2_start_time.drop(columns=['Week_Type'], inplace=True)  # 임시로 만든 Week_Type 컬럼 삭제

        return simul_trip_sequence_2_start_time
    
    # Assign start times to the trips with previously assigned dwell times
    result_table = assignStartTime(dwell_table, startTime_dict, print_progress)
    
    return result_table


# ## 4. Trip Mode Assigner

# In[10]:


def put_tripMode(df, trip_mode_origin, print_progress=True):
    """
    Assigns a travel mode to each trip in the simulated dataset based on the mode distribution from original data,
    considering the age group and trip purpose. It also adjusts the assigned trip modes to reflect realistic travel patterns,
    such as ensuring round trips have consistent modes and modifying modes based on trip sequence.

    Parameters:
    - df: DataFrame containing the simulated trip data.
    - trip_mode_origin: DataFrame with the original distribution of trip modes by age class and trip purpose.
    - print_progress: Boolean indicating whether to display progress during the execution.

    Returns:
    - A DataFrame with an assigned 'Trip_mode' for each trip, adjusted for realism based on trip sequences.
    """
    
    if print_progress == True:
        print("<Trip mode assigner>")
        
    tqdm.pandas(desc="1. Assign initial trip modes")
    
    # Convert the original trip mode distributions into a dictionary for efficient lookup
    probability_dict = trip_mode_origin.set_index(['age_class', 'Trip_pur', 'Trip_mode'])['Trip_modeP'].to_dict()

    def assign_trip_mode(row):
        """
        Samples a trip mode based on the distribution for the given age group and trip purpose.
        """
        age = row['ageGroup']
        purpose = row['TRPPURP']

        # Extract mode probabilities for the given age and purpose, defaulting to 0 if not found
        mode_probabilities = {mode: probability_dict.get((age, purpose, mode), 0) for mode in trip_mode_origin['Trip_mode'].unique()}

        # Sample a mode based on the probabilities
        return np.random.choice(list(mode_probabilities.keys()), p=list(mode_probabilities.values()))

    if print_progress == True:
        df['Trip_mode'] = df.progress_apply(assign_trip_mode, axis=1)
    else:
        df['Trip_mode'] = df.apply(assign_trip_mode, axis=1)
    
    tqdm.pandas(desc="2. Modify Trip_mode based on the first row of each group")

    def modify_trip_mode(group):        
        """
        Modifies the trip mode of the first trip in each group if its purpose is not 'Home',
        since the mode for trips starting from 'Home' might follow a different pattern.
        """
        
        first_row = group.iloc[0]
        if first_row['TRPPURP'] == 'Home':
            group.at[first_row.name, 'Trip_mode'] = np.nan # Clear the mode if the first trip starts from 'Home'
        else:
            print(first_row['uniqID'], first_row['Day_Type'], 'Ah uh') # Log if the first trip doesn't start from 'Home'
        return group

    # Apply mode modification to each group of trips
    if print_progress == True:
        put_trip_mode_df = df.groupby(['uniqID', 'Day_Type']).progress_apply(modify_trip_mode)
    else:
        put_trip_mode_df = df.groupby(['uniqID', 'Day_Type']).apply(modify_trip_mode)
    
    
    def adjust_trip_mode_for_car(df):
        """
        Adjusts the trip mode to 'Car' for the last trip if the first trip mode is 'Car',
        reflecting the assumption that round trips typically use the same mode.
        It also checks for consistency in modes for intermediate trips.
        """
    
        tqdm.pandas(desc="3. Adjust Trip_mode for round / one-way trip")
    
        def process_group(group):
            if len(group) == 1:
                return group # No adjustment needed for single-trip groups

            if group['Trip_mode'].iloc[1] != 'Car':
                return group # No adjustment if the first trip mode isn't 'Car'

            # 첫 Trip_mode가 Car라면 마지막 Trip_mode를 Car로 설정
            group['Trip_mode'].iloc[-1] = 'Car'

            # Set the last trip mode to 'Car' if the first is 'Car'
            current_mode = 'Car'
            for idx in range(2, len(group) - 1):
                if group['Trip_mode'].iloc[idx] != current_mode:
                    # Check for round trips and adjust modes accordingly
                    next_indices = range(idx+1, len(group))
                    if group['TRPPURP'].iloc[idx-1] in [group['TRPPURP'].iloc[next_idx] for next_idx in next_indices]:
                        current_mode = group['Trip_mode'].iloc[idx]
                    else:
                        # Adjust mode for one-way trips
                        group['Trip_mode'].iloc[idx] = current_mode

            return group
        
        # Apply the adjustment to all groups
        if print_progress == True:
            df = df.groupby(['uniqID', 'Day_Type']).progress_apply(process_group).reset_index(drop=True)
        else:
            df = df.groupby(['uniqID', 'Day_Type']).apply(process_group).reset_index(drop=True)
        return df

    adjust_trip_mode_df = adjust_trip_mode_for_car(put_trip_mode_df)
    
    return adjust_trip_mode_df


# ## 5. Spatial Trip Route Estimator
# ### 5.0. Data staging (1): ratio between straight path and network path

# In[11]:


def random_point_within(polygon):
    """
    Generates a random point within a given polygon.

    Parameters:
    - polygon: The shapely Polygon object representing an area, such as a census block group (CBG).

    Returns:
    - A shapely Point object that lies within the polygon.
    """
    minx, miny, maxx, maxy = polygon.bounds # Get the bounding box of the polygon.
    while True:
        # Generate a random point within the bounding box.
        p = Point(random.uniform(minx, maxx), random.uniform(miny, maxy))
        # Check if the generated point is within the actual polygon.
        if polygon.contains(p):
            return p # Return the point if it's inside the polygon

def calculate_sampled_network_distance(cbg_gdf, network_road, num_samples=100):
    """
    Calculates the average ratio between network (road) distance and straight-line distance for randomly selected pairs of points within CBGs.

    Parameters:
    - cbg_gdf: GeoDataFrame containing the geometries of census block groups (CBGs).
    - network_road: The graph representing the road network, typically from the osmnx library.
    - num_samples: The number of random pairs of points to sample for the calculation.

    Returns:
    - A list of distance ratios for all valid samples and the average ratio across these samples.
    """
    distance_ratios = [] # Store the ratio of network to straight distance for each sample.

    for _ in tqdm(range(num_samples), desc="Calculating Sampled Distances"):
        try:
            # Randomly select two CBGs from the GeoDataFrame.
            sampled_cbg = cbg_gdf.sample(n=2).reset_index(drop=True)
            cbg_1 = sampled_cbg.loc[0]
            cbg_2 = sampled_cbg.loc[1]
#             print(cbg_1, cbg_2)

            # Generate random points within each selected CBG.
            point_1 = random_point_within(cbg_1.geometry)
            point_2 = random_point_within(cbg_2.geometry)

            # Find the nearest points on the road network to the generated points.
            nearest_road_point_1 = ox.distance.nearest_nodes(network_road, point_1.x, point_1.y)
            nearest_road_point_2 = ox.distance.nearest_nodes(network_road, point_2.x, point_2.y)

            # Calculate the network distance between these nearest road points.
            network_distance = nx.shortest_path_length(network_road, nearest_road_point_1, nearest_road_point_2, weight='length') / 1000
#             print('network_distance: ', network_distance)

            # Calculate the straight-line (geodesic) distance between the generated points.
            coords_1 = (point_1.y, point_1.x)
            coords_2 = (point_2.y, point_2.x)
            
            straight_distance = geodesic(coords_1, coords_2).meters / 1000
            
            # Calculate the ratio of network to straight distance, if straight distance is not zero.
            if straight_distance != 0:
                distance_ratio = network_distance / straight_distance
                distance_ratios.append(distance_ratio)

        except Exception as e: # Handle any errors during the process.
            print("Error:", e)
            
            continue

    # Calculate and return the average distance ratio if any valid ratios were calculated.
    if distance_ratios:
        average_distance_ratio = np.mean(distance_ratios)
        
        return distance_ratios, average_distance_ratio
    else:
        #print("No valid distance ratios calculated.")
        return None


# ### 5.1. Estimate probabilistic destinations for trip purpose t

# In[12]:


def choose_destination(probabilities):
    """
    Selects a destination based on a set of given probabilities for each potential destination.

    Parameters:
    - probabilities: A dictionary where keys are destination identifiers (e.g., CBG IDs) and
                     values are the corresponding probabilities of choosing each destination.

    Returns:
    - A selected destination identifier, chosen based on the provided probabilities.
    """
    
    destinations = list(probabilities.keys()) # Extract a list of potential destinations
    probabilities = list(probabilities.values())  # Extract the corresponding list of probabilities for each destination
    probabilities_sum = sum(probabilities)  # Calculate the sum of all probabilities for normalization
    normalized_probabilities = [p / probabilities_sum for p in probabilities]  # Normalize probabilities to ensure they sum to 1
    
    return np.random.choice(destinations, p=normalized_probabilities) # Randomly choose a destination based on the normalized probabilities



def assign_sdr_destination(row, assigned_trip_table, prob_df, ageGroup, day_type):
    """
    Assigns a destination for trips with specific requirements based on the trip's age group and day type.
    This function is particularly tailored for 'S_d_r' (School, daily care, and religion) trips, determining
    possible destinations and selecting one based on predefined probabilities.

    Parameters:
    - row: The current row of the DataFrame being processed, representing a single trip.
    - assigned_trip_table: A DataFrame tracking previously assigned destinations to ensure consistency.
    - prob_df: A DataFrame containing probabilities for various destinations based on different conditions.
    - ageGroup: The age group of the individual undertaking the trip.
    - day_type: The type of day on which the trip occurs ('weekday' or 'weekend').

    Returns:
    - Dest: The chosen destination based on the trip's conditions.
    - TRPPURP_det: A detailed trip purpose, potentially refined during the destination assignment process.
    - assigned_trip_table: The updated DataFrame of assigned destinations, including the current trip's assignment.
    """
    
    ws_wd = (row['Ws'], row['Wd']) # Extract workplace and dwelling weights for the trip
    prob_row = prob_df[(prob_df['area'] == row['Home_cbg']) & (prob_df['Ws'] == ws_wd[0]) & (prob_df['Wd'] == ws_wd[1])]

    if prob_row.empty:
        return None, None, assigned_trip_table # If no matching probability row, return None for destination and purpose

    # 1. Determine possible destinations based on age group and day type
    if ageGroup == 'Child':
        possible_destinations = ['Religion'] if day_type == 'weekend' else ['School', 'Dailycare']
    elif ageGroup == 'Teen':
        possible_destinations = ['Religion'] if day_type == 'weekend' else ['School']
    elif ageGroup == 'Adult':
        possible_destinations = ['University', 'Dailycare'] if day_type == 'weekday' else ['Dailycare', 'Religion']
    else:
        possible_destinations = ['Dailycare', 'Religion']

            
    # 2. Calculate probabilities for each possible destination within the chosen category
    category_probabilities = {}
    for destination in possible_destinations:
        prob_col = f'{day_type}_{destination}'
        if prob_col in prob_row.columns:
            category_probabilities[destination] = eval(prob_row[prob_col].values[0])
        else:
            category_probabilities[destination] = 0.0

            
    # 3. Select a destination category from the available options
    chosen_category = np.random.choice(list(category_probabilities.keys()))
#     print(chosen_category)
    
    # 4. Further logic to select a specific location within the chosen destination category
    if not category_probabilities[chosen_category]:
        if chosen_category == 'University':
            chosen_category = 'Dailycare'
            if not category_probabilities[chosen_category]:
                # If there are no destinations for 'University' 'Religion' trip
                return None, chosen_category, assigned_trip_table
        else:
            # Ptobability distribution corresponding to the selected category is empty
            return None, chosen_category, assigned_trip_table

    
    # 5. Update the assigned_trip_table with the chosen destination
    if ageGroup == 'Child':
        assigned_rows = assigned_trip_table[(assigned_trip_table['uniqID'] == row['uniqID']) & (assigned_trip_table['TRPPURP'].isin(['School', 'Dailycare']))]
        if not assigned_rows.empty:
            # If one of the School or Dailycare is selected, return only TRPPURP
            assigned_category = assigned_rows.iloc[0]['TRPPURP']
            return assigned_rows.iloc[0]['area'], f'{assigned_category}', assigned_trip_table
    
    else:
        assigned_row = assigned_trip_table[(assigned_trip_table['uniqID'] == row['uniqID']) & (assigned_trip_table['TRPPURP'] == chosen_category)]
        if not assigned_row.empty:
            # If there is assigned destination, return that destination
            return assigned_row.iloc[0]['area'], f'{assigned_row.iloc[0]["TRPPURP"]}', assigned_trip_table
    
    
    areas, probabilities = zip(*category_probabilities[chosen_category].items())
    
    probabilities = np.array(probabilities) / sum(probabilities)  # Normalize Prob
#     print(probabilities)
    chosen_destination = np.random.choice(areas, p=probabilities)
#     print("asd", chosen_destination)

    # 6. Add selected destination and location to assigned_trip_table
    
    if chosen_destination:
        new_row = pd.DataFrame({'uniqID': [row['uniqID']], 'ageGroup': [ageGroup], 'TRPPURP': [chosen_category], 'area': [chosen_destination]})
        assigned_trip_table = pd.concat([assigned_trip_table, new_row], ignore_index=True)
    
    # 7. Update TRPPURP_det
    
    return chosen_destination, chosen_category, assigned_trip_table




def set_origin_destination(simul_df):
    """
    Iteratively sets the 'Origin' for each trip in the simulated dataframe. 
    The origin of a trip is determined based on the destination of the previous trip for the same individual on the same day.
    Special handling is applied to the first trip of the day and to trips with specific purposes.

    Parameters:
    - simul_df: DataFrame containing the simulated trip data with destinations ('Dest') already assigned.

    Returns:
    - The modified DataFrame with the 'Origin' column added and populated based on the logic outlined.
    """
    
    # Copy the DataFrame to avoid modifying the original data in place
    simul_df = simul_df.copy()
    # Initialize the 'Origin' column with NaN values
    simul_df['Origin'] = np.nan
    
    # Group the DataFrame by unique identifier and day type to process each individual's trips per day separately
    for (uniqID, day_type), group in simul_df.groupby(['uniqID', 'Day_Type']):
        prev_dest = None
        for idx, row in group.iterrows():
            if pd.isna(prev_dest):  # Check if this is the first trip of the day
                
                # For the first trip, set the origin as the home location if the trip purpose is 'Home'
                if row['TRPPURP'] == 'Home':
                    simul_df.at[idx, 'Origin'] = row['Home_cbg']
                else:
                    # If the first trip's purpose is not 'Home', leave the origin as NaN
                    simul_df.at[idx, 'Origin'] = None 
            else:
                # For subsequent trips, set the origin as the destination of the previous trip
                simul_df.at[idx, 'Origin'] = prev_dest  
                
            # Update prev_dest with the current trip's destination for the next iteration    
            if pd.notna(row['Dest']):  
                prev_dest = row['Dest']
    
    return simul_df




def assignDest(prob_df, simul_df, W_d_W_s = False, print_progress = True):
    """
    Estimates spatial trip routes by assigning origins and destinations for simulated trips. 
    This process uses probabilities derived from real-world data to make the simulated trips more realistic in terms of spatial distribution. 

    Parameters:
    - prob_df: DataFrame containing probability distributions for various destinations based on certain conditions.
    - simul_df: DataFrame containing the simulated trip data.
    - W_d_W_s: Either a boolean flag indicating whether custom weights/distributions should be used, 
             or a dictionary mapping trip purposes to specific weight/distribution combinations.
    - print_progress: Boolean indicating whether to display progress information during the execution.

    Returns:
    - simul_df: Updated DataFrame with 'Origin' and 'Dest' fields assigned based on the estimated routes.
    - assigned_trip_table: DataFrame tracking assigned destinations for validation and further analysis.
    """
    
    if print_progress == True:
        print('<spatial trip route estimator>')        
        
    # Initialize an empty DataFrame to track assigned destinations
    assigned_trip_table = pd.DataFrame(columns=['uniqID', 'ageGroup', 'TRPPURP', 'area'])
    
    # Ensure 'area' column in prob_df is of string type for consistency in key matching
    prob_df['area'] = prob_df['area'].astype('str')
    
    def transform_value(x):
        if isinstance(x, float):  # If the value is a float
            return f"{x:.0f}"  # Convert to string without decimal
        elif isinstance(x, int):  # If the value is an integer
            return str(x)  # Convert to string
        elif isinstance(x, str):  # If the value is already a string
            return x  # Keep it as is
        else:
            return x  # For other types, return as is

    # Apply the transformation to the 'Home_cbg' and 'cbg' columns
    simul_df['Home_cbg'] = simul_df['Home_cbg'].apply(transform_value)
    
    if W_d_W_s == False:
        wd_ws_combinations = prob_df[['Ws', 'Wd']].drop_duplicates().to_dict('records')
#         print(wd_ws_combinations)
        trppurp_ws_wd_mapping = {trppurp: random.choice(wd_ws_combinations) for trppurp in simul_df['TRPPURP'].unique() if trppurp != 'Home'}
    else:
        trppurp_ws_wd_mapping = W_d_W_s
        
    # Prepare simul_df by initializing necessary columns for processing
    simul_df['Origin'] = np.nan
    simul_df['Dest'] = np.nan
    simul_df['Ws'] = simul_df['TRPPURP'].map(lambda x: trppurp_ws_wd_mapping.get(x, {}).get('Ws'))
    simul_df['Wd'] = simul_df['TRPPURP'].map(lambda x: trppurp_ws_wd_mapping.get(x, {}).get('Wd'))
    
    trppurp_to_prob_col = {
        'Work': 'Work',
        'Serv_trip': 'Serv_trip',
        'Meals': 'Meals',
        'Rec_lei': 'Rec_lei',
        'V_fr_rel': 'V_fr_rel',
        'D_shop': 'D_shop',
        'Others': 'Others'
    }
    
    simul_df['TRPPURP_det'] = simul_df['TRPPURP'] # new column: TRPPURP_det - detailed TRPPURP
    
     # Process each trip to assign destinations based on the specified probabilities and conditions
    iterator = tqdm(simul_df.groupby(['uniqID', 'ageGroup']), desc = '1. Assign trip-routes (Origin and Dest)') if print_progress else simul_df.groupby(['uniqID', 'ageGroup'])
    
    for (uniqID, ageGroup), group in iterator:
       
        prev_Dest = None
        main_workplace_assigned = False
        for idx, row in group.iterrows():
            # Special handling for trips with the 'Home' purpose
            if row['TRPPURP'] == 'Home':  # Home -> Home_cbg
                simul_df.at[idx, 'Dest'] = row['Home_cbg']
                
                continue
                
            # Additional logic to handle trips with specific purposes such as 'S_d_r'    
            if row['TRPPURP'] == 'S_d_r': # S_d_r -> Complex condition
                day_type = 'weekend' if row['Day_Type'] in ['Saturday', 'Sunday'] else 'weekday'
                Dest, TRPPURP_det, assigned_trip_table = assign_sdr_destination(row, assigned_trip_table, prob_df, ageGroup, day_type)
                
                simul_df.at[idx, 'Dest'] = Dest
                simul_df.at[idx, 'TRPPURP_det'] = TRPPURP_det
                continue
            

            day_type = 'weekend' if row['Day_Type'] in ['Saturday', 'Sunday'] else 'weekday'
            trppurp = trppurp_to_prob_col.get(row['TRPPURP'])
            
            if trppurp is not None:
                ws_wd = (row['Ws'], row['Wd'])
                prob_row = prob_df[(prob_df['area'] == row['Home_cbg']) & (prob_df['Ws'] == ws_wd[0]) & (prob_df['Wd'] == ws_wd[1])]

                if not prob_row.empty:
#                     print(row['TRPPURP'])
                    if row['TRPPURP'] == 'Work':
                        if main_workplace_assigned:
                            # Sub workplace: Assign Dest based on prob
                            prob_col = f'{day_type}_{trppurp}'
                            probabilities = eval(prob_row[prob_col].values[0])
#                             print("asdasd", probabilities)
                            Dest = choose_destination(probabilities)
                        else:
                            # Main workplace Assign first destination for Work trip and save
                            prob_col = f'{day_type}_{trppurp}'
                            probabilities = eval(prob_row[prob_col].values[0])
                            Dest = choose_destination(probabilities)
                            assigned_trip_table = assigned_trip_table.append({'uniqID': uniqID, 'ageGroup': ageGroup, 'TRPPURP': 'Work', 'area': Dest}, ignore_index=True)
                            main_workplace_assigned = True
                    elif row['TRPPURP'] == 'D_shop':
                        # D_shop: Large_shop at 80% probability, Etc_shop at 20% probability
                            
                        shop_choice = np.random.choice(['Large_shop', 'Etc_shop'], p=[0.8, 0.2])
                        simul_df.at[idx, 'TRPPURP_det'] = shop_choice  # Update TRPPURP_det
                        prob_col = f'{day_type}_{shop_choice}'
                        probabilities = eval(prob_row[prob_col].values[0])
                        Dest = choose_destination(probabilities)
                        
                    else:
                        # Non-Work TRPPURP: Assign Dest based on prob
                        prob_col = f'{day_type}_{trppurp}'
                        probabilities = eval(prob_row[prob_col].values[0])
                        Dest = choose_destination(probabilities)
                        

                    
                    simul_df.at[idx, 'Dest'] = Dest


    # Update 'Origin' based on the assigned 'Dest' values
    simul_df = set_origin_destination(simul_df)
    return simul_df, assigned_trip_table


# ### 5.2. Compute trip distance and duration

# In[13]:


# 거리 구하기 

def estimate_tripDist_Time(simul_df, cbg_gdf, distance_ratio = 1.2229481, print_progress = True):
    """
    Estimates the distances and travel times for each trip in the simulated dataset. 
    This function uses the geographic information from census block groups (CBG) and a predefined distance ratio to calculate network distances. 
    It then applies average speeds based on the mode of transport and the age group of the traveler to estimate travel times.

    Parameters:
    - simul_df: DataFrame containing simulated trip data with origins and destinations specified by CBG codes.
    - cbg_gdf: GeoDataFrame containing CBG polygons and their FIPS codes for locating origins and destinations.
    - distance_ratio: The average ratio between straight-line distances and network distances. 
                       This is used to convert geodesic distances to more realistic network distances.
    - print_progress: Boolean flag indicating whether to display a progress bar during execution.

    Returns:
    - Updated simul_df with estimated trip distances ('TripDist') in kilometers and trip times ('TripTime') in minutes.
    """

    # Function to generate a random point within a given polygon. This is used to simulate starting
    # and ending positions within the origin and destination CBGs.
    def random_point_within(polygon):
        # Extract the bounds of the polygon to define the area for generating random points
        minx, miny, maxx, maxy = polygon.bounds
        while True:
            p = Point(random.uniform(minx, maxx), random.uniform(miny, maxy))
            # Check if the point is within the polygon itself, not just the bounding box
            if polygon.contains(p):
                return p

    # Function to estimate network distances between origin and destination CBGs
    def estimate_network_distance(simul_df, cbg_gdf, distance_ratio):
        simul_df['TripDist'] = None # Initialize the column for storing trip distances

        # Iterate over each row in the dataframe to calculate distances
        iterator = tqdm(simul_df.iterrows(), total=simul_df.shape[0], desc='2. Estimating trip distances') if print_progress else simul_df.iterrows()
        
        for idx, row in iterator:
            # Retrieve the polygons for the origin and destination CBGs
            origin_cbg = cbg_gdf[cbg_gdf['FIPS_BLKGR'] == str(int(row['Origin']))].geometry.iloc[0]
            dest_cbg = cbg_gdf[cbg_gdf['FIPS_BLKGR'] == str(int(row['Dest']))].geometry.iloc[0]

            # Generate random points within the origin and destination polygons
            origin_point = random_point_within(origin_cbg)
            dest_point = random_point_within(dest_cbg)


            # Calculate the geodesic (straight-line) distance between the two points
            coords_1 = (origin_point.y, origin_point.x)
            coords_2 = (dest_point.y, dest_point.x)

            straight_distance = geodesic(coords_1, coords_2).meters / 1000

            # Estimate the network distance using the provided distance ratio
            estimated_network_distance = straight_distance * distance_ratio

            # Store the calculated distance in the dataframe
            simul_df.at[idx, 'TripDist'] = estimated_network_distance

        return simul_df
    
    # Estimate the network distances for the trips
    trip_dist_df = estimate_network_distance(simul_df, cbg, distance_ratio)
    
    # Calculate trip times based on the mode of transport and the traveler's age group
    def calculate_trip_time(row):
        speed = np.nan

        if row['Trip_mode'] == 'Walk':
            if row['ageGroup'] == 'Child' or row['ageGroup'] == 'Teen':
                speed = 4.82
            elif row['ageGroup'] == 'Adult':
                speed = np.random.uniform(4.54, 4.82)
            elif row['ageGroup'] == 'MidAdult':
                speed = np.random.uniform(4.43, 4.54)
            elif row['ageGroup'] == 'Seniors':
                speed = np.random.uniform(3.42, 4.34)
        elif row['Trip_mode'] == 'Car':
            speed = 39.74
        elif row['Trip_mode'] == 'PTrans':
            speed = 18.79
        elif row['Trip_mode'] == 'Bicy':
            speed = 7.72

        # Calculate the trip time in minutes based on the distance and speed
        trip_time = round((row['TripDist'] / speed) * 60, 0)
        if trip_time < 1: trip_time = 1

        # Convert inf into NaN
        if np.isinf(trip_time):
            return np.nan

        return trip_time
    
    # Initialize the 'TripTime' column and calculate trip times for each row
    trip_dist_df['TripTime'] = None

    # Calculate 'TripTime' for each raw
    iterator = tqdm(trip_dist_df.iterrows(), total=trip_dist_df.shape[0], desc="3. Calculating Trip Time") if print_progress else trip_dist_df.iterrows()
    
    for idx, row in iterator:
        trip_dist_df.at[idx, 'TripTime'] = calculate_trip_time(row)
        
        
    # Final step: Calculate start and end times for each trip and organize the DataFrame columns.

    # Integrating tqdm progress bar with pandas operations for visual progress feedback.
    tqdm.pandas(desc="4. Calculating Start and End Times")

    def calculate_start_end_times(group):
        """
        Calculates start and end times for each trip within a group of trips by the same individual on the same day.
        Adjusts the 'Dwell_Time' for the first trip and sets start and end times based on the sequence of trips.

        Parameters:
        - group: Grouped DataFrame segment representing all trips for an individual on a specific day.

        Returns:
        - The group with updated start and end times for each trip.
        """
        
        # Handle cases where a group has only one trip.
        if len(group) == 1:
            group['TripSTARTT'] = np.nan
            group['TripENDT'] = np.nan
            group['start_min'] = np.nan
            group['end_min'] = np.nan
            return group

        # Sort trips within the group based on their sequence.
        group = group.sort_values(by='sequence')

        # Special handling for the first trip, setting 'Dwell_Time' based on the second trip's start time.
        if group.iloc[0]['TRPPURP'] == 'Home':
            group.at[group.index[0], 'Dwell_Time'] = group.iloc[1]['sta_T_min']

        # Initialize columns for start and end times.
        group['TripSTARTT'] = np.nan
        group['TripENDT'] = np.nan
        group['start_min'] = 0
        group['end_min'] = group.iloc[0]['Dwell_Time']

        # Iterate over trips to calculate and set start and end times.
        for i in range(1, len(group)):
            prev_row = group.iloc[i - 1]
            prev_start = 0 if pd.isna(prev_row['TripSTARTT']) else prev_row['TripSTARTT']
            prev_dwell = 0 if pd.isna(prev_row['Dwell_Time']) else prev_row['Dwell_Time']
            prev_trip_time = 0 if pd.isna(prev_row['TripTime']) else prev_row['TripTime']
            prev_end = 0 if pd.isna(prev_row['TripENDT']) else prev_row['TripENDT']
            prev_end_D_min = 0 if pd.isna(prev_row['end_min']) else prev_row['end_min']

            current_dwell = group.at[group.index[i], 'Dwell_Time'] if not pd.isna(group.at[group.index[i], 'Dwell_Time']) else 0

            # Calculate the start time for the current trip.
            group.at[group.index[i], 'TripSTARTT'] = prev_start + prev_dwell + prev_trip_time
            group.at[group.index[i], 'TripENDT'] = group.at[group.index[i], 'TripSTARTT'] + group.at[group.index[i], 'TripTime']
            
            # Update start and end minutes for the next iteration.
            group.at[group.index[i], 'start_min'] = group.at[group.index[i], 'TripENDT']
            group.at[group.index[i], 'end_min'] = group.at[group.index[i], 'start_min'] + current_dwell

        return group

    # Apply the function to each group of trips by the same individual on the same day.
    if print_progress:
        trip_dist_df = trip_dist_df.groupby(['uniqID', 'Day_Type'], group_keys=False).progress_apply(calculate_start_end_times)
    else:
        trip_dist_df = trip_dist_df.groupby(['uniqID', 'Day_Type'], group_keys=False).apply(calculate_start_end_times)
    
    # Clean up: Remove unnecessary columns and rearrange the remaining ones.
    if 'sta_T_min' in trip_dist_df.columns:
        trip_dist_df.drop('sta_T_min', axis=1, inplace=True)
    if 'Unnamed: 0' in trip_dist_df.columns:
        trip_dist_df.drop('Unnamed: 0', axis=1, inplace=True)
    if 'trip_count_class'in trip_dist_df.columns:
        trip_dist_df.drop('trip_count_class', axis=1, inplace=True)

    # Define the desired column order for the final DataFrame.
    desired_order = [
        'TRPPURP', 'TRPPURP_det', 'Ws', 'Wd', 'Origin', 'Dest', 
        'TripDist', 'Trip_mode', 'TripSTARTT', 'TripTime', 'TripENDT', 
        'start_min', 'Dwell_Time', 'end_min'
    ]

    current_columns = trip_dist_df.columns.tolist()

    # Ensure the DataFrame follows the desired column order, moving any unspecified columns to the end.
    new_order = [col for col in current_columns if col not in desired_order] + desired_order

    trip_dist_df = trip_dist_df[new_order]

    
    return trip_dist_df


# ### 5.3. Optimize Trips with Logical and space-time constraints

# In[14]:


def optimizeTrips_constraint(df, dwellTime_dict, print_progress = True):
    """
    Applies constraints to optimize individual trips by adjusting start and end times and updating trip modes based on certain logical conditions, 
    such as changing modes from walking to driving or public transport for longer trips.

    Parameters:
    - df: DataFrame containing simulated trip data.
    - dwellTime_dict: Dictionary containing dwell time distributions for various scenarios.
    - print_progress: Boolean indicating whether to show progress during execution.

    Returns:
    - Updated DataFrame with optimized trips.
    """

    def update_rows(group_df):
        """
        Updates the start and end times for trips within a group based on dwell times and trip durations.
        
        Parameters:
        - group_df: DataFrame segment representing trips for an individual on a specific day.
        
        Returns:
        - The updated group DataFrame with recalculated start and end times.
        """
        
        # Update the first row's end time based on its dwell time
        group_df.iloc[0, group_df.columns.get_loc('end_min')] = group_df.iloc[0]['Dwell_Time']

        # Iterate through the rest of the rows to update start and end times sequentially
        for i in range(1, len(group_df)):
            group_df.iloc[i, group_df.columns.get_loc('TripSTARTT')] = group_df.iloc[i-1]['end_min']
            group_df.iloc[i, group_df.columns.get_loc('TripENDT')] = group_df.iloc[i]['TripSTARTT'] + group_df.iloc[i]['TripTime']
            group_df.iloc[i, group_df.columns.get_loc('start_min')] = group_df.iloc[i]['TripENDT']
            group_df.iloc[i, group_df.columns.get_loc('end_min')] = group_df.iloc[i]['start_min'] + group_df.iloc[i]['Dwell_Time']

        return group_df


    def calculate_trip_time(row):
        """
        Calculates the trip time based on the trip mode and age group, adjusting speeds accordingly.
        
        Parameters:
        - row: A row of the DataFrame representing a single trip.
        
        Returns:
        - The calculated trip time in minutes.
        """
        speed = np.nan

        # Define speed values based on trip mode and age group
        if row['Trip_mode'] == 'Walk':
            if row['ageGroup'] == 'Child' or row['ageGroup'] == 'Teen':
                speed = 4.82
            elif row['ageGroup'] == 'Adult':
                speed = np.random.uniform(4.54, 4.82)
            elif row['ageGroup'] == 'MidAdult':
                speed = np.random.uniform(4.43, 4.54)
            elif row['ageGroup'] == 'Seniors':
                speed = np.random.uniform(3.42, 4.34)
        elif row['Trip_mode'] == 'Car':
            speed = 39.74 # Fixed speed for car travel
        elif row['Trip_mode'] == 'PTrans':
            speed = 18.79
        elif row['Trip_mode'] == 'Bicy':
            speed = 7.72

        # Calculate trip time (distance/speed), then convert to minutes and round
        trip_time = round((row['TripDist'] / speed) * 60, 0)
        if trip_time < 1: trip_time = 1

        # Convert infinity values to NaN
        if np.isinf(trip_time):
            return np.nan

        return trip_time

    def update_group(group):
        """
        Updates the trip mode for each trip in the group based on the previous trip's mode and the current trip's duration.
        
        Parameters:
        - group: DataFrame segment representing trips for an individual on a specific day.
        
        Returns:
        - The updated group DataFrame with modified trip modes.
        """
        
        # If the current trip mode is 'Walk' and the trip time is >= 60 minutes
        for idx in group.index[1:]:  # Iterate through the group to check and update the trip mode
            if group.loc[idx, 'Trip_mode'] == 'Walk' and group.loc[idx, 'TripTime'] >= 60:
                # If the previous trip's mode was 'Car', set the current trip's mode to 'Car'
                if group.loc[group.index[group.index.get_loc(idx) - 1], 'Trip_mode'] == 'Car':
                    group.loc[idx, 'Trip_mode'] = 'Car'
                else:
                    # Otherwise, change the mode to 'PTrans' (Public Transport)
                    group.loc[idx, 'Trip_mode'] = 'PTrans'

                # Recalculate the trip time with the updated mode
                group.loc[idx, 'TripTime'] = calculate_trip_time(group.loc[idx])

        return group

    def logicalConstraint(df):
        """
        Applies a logical constraint to update trip modes based on the previous trip's mode
        and the duration of the current trip.

        Parameters:
        - df: DataFrame of the simulated trip data.

        Returns:
        - The DataFrame with updated trip modes where necessary.
        """
        updated_df = df.copy()
        
        # Generate lists of Day_Type and uniqID that stastify condition by filtering raws
        condition = (updated_df['Trip_mode'] == 'Walk') & (updated_df['TripTime'] >= 60)
        unique_pairs = updated_df[condition][['uniqID', 'Day_Type']].drop_duplicates()

        iterator = tqdm(unique_pairs.iterrows(), total = len(unique_pairs), desc = '5. Logical constraint (trip mode)' ) if print_progress else unique_pairs.iterrows()
        
        # Apply the update_group function to each group of trips by the same individual on the same day
        for _, row in iterator:
            group_data = updated_df.loc[(updated_df['uniqID'] == row['uniqID']) & (updated_df['Day_Type'] == row['Day_Type'])]
            updated_group = update_group(group_data)

    #         display(updated_group)

            updated_group = update_rows(updated_group)
            # Update the main DataFrame with the modified group
            updated_df.update(updated_group)

        return updated_df

    #-----------------------
    #-----------------------
    #-----------------------
    # Apply logical constraints to optimize trip modes
    updated_df = logicalConstraint(df)
    #-----------------------
    #-----------------------
    #-----------------------

    def assignDwellTime(df, distribution_list, print_ = True):
        """
        Assigns dwell times to trips in the dataset based on predefined distributions. The function updates the
        'Dwell_Time' column by sampling from distributions specific to each trip's characteristics, including the
        trip's day type, age group, and purpose.

        Parameters:
        - df: DataFrame containing the simulated trip data.
        - distribution_list: A dictionary or other data structure containing dwell time distributions for various trip scenarios.
        - print_: Boolean indicating whether to show progress during the execution.

        Returns:
        - DataFrame with updated 'Dwell_Time' for each trip.
        """
        
        # Copy the DataFrame to avoid modifying the original data in place.
        simul_trip_sequence = df.copy()

        tqdm.pandas(desc="1. classify tripCount of simulated data")

        # Calculate the size of each group (by 'uniqID' and 'Day_Type') to determine the trip count class.
        group_counts = simul_trip_sequence.groupby(['uniqID', 'Day_Type']).size().to_dict()
        
        def assign_class(row):
            group_size = group_counts[(row['uniqID'], row['Day_Type'])]

            if group_size <= 3:
                return 1 # Class 1 for groups with up to 3 trips
            elif 4 <= group_size <= 5:
                return 2 # Class 2 for groups with 4-5 trips
            else:
                return 3

        # Apply the classification function to each row.
        if print_ == True:
            simul_trip_sequence['trip_count_class'] = simul_trip_sequence.progress_apply(assign_class, axis=1)
        else:
            simul_trip_sequence['trip_count_class'] = simul_trip_sequence.apply(assign_class, axis=1)


        # 2. Sampling Dwell_Time from distribution
        tqdm.pandas(desc="2. Assign Dwell time")

        def assign_dwell_time(row):
            # Convert the 'Day_Type' to a more generic form for lookup.
            day_type_map = {
                'Monday': 'Weekday',
                'Tuesday': 'Weekday',
                'Wednesday': 'Weekday',
                'Thursday': 'Weekday',
                'Friday': 'Weekday',
                'Saturday': 'Weekend',
                'Sunday': 'Weekend'
            }
            day_type = day_type_map[row['Day_Type']]

            # Construct a key for the distributions based on the row's characteristics
            key = (row['ageGroup'], day_type, row['trip_count_class'], row['TRPPURP'])
            
            # Retrieve the distribution for the given key, defaulting to [0] if not found.
            dwell_times = distribution_list.get(key, [0])

            # Handle cases where the distribution is missing or empty.
            if not isinstance(dwell_times, (list, np.ndarray)) or len(dwell_times) == 0:
                # Sample a dwell time from the distribution.
                dwell_time_sample = np.random.randint(10, 301)
                print(f"cannot find from dic, put random dwelltime...({row['ageGroup']}, {row['Day_Type']}, {row['trip_count_class']}, {row['TRPPURP']}, value: {dwell_time_sample})")
                return dwell_time_sample

            dwell_time_sample = -1
            while dwell_time_sample < 0:
                dwell_time_sample = np.random.choice(dwell_times)

            return dwell_time_sample

        # Apply the dwell time assignment function to each row.
        if print_ == True:
            simul_trip_sequence['Dwell_Time'] = simul_trip_sequence.progress_apply(assign_dwell_time, axis=1)
        else:
            simul_trip_sequence['Dwell_Time'] = simul_trip_sequence.apply(assign_dwell_time, axis=1)

        # drop column seq_NHTS
        if 'seq_NHTS' in simul_trip_sequence.columns:
            simul_trip_sequence = simul_trip_sequence.drop('seq_NHTS', axis=1)


        return simul_trip_sequence
    
       
    # If time schedule for 1 day is too long, adjust the time. if > 2400 of end_min -> adjust end_min below 2400
    def timeConstraint(df, dwellTime_dict, assignDwellTime, print_progress):
        """
        Adjusts the dwell times for each trip to ensure that the total time spent on all daily activities does not exceed 24 hours.
        This function iteratively adjusts dwell times using a predefined distribution and recalculates trip start and end times 
        to meet the time constraint.

        Parameters:
        - df: DataFrame containing the simulated trip data with preliminary 'Dwell_Time', 'TripTime', etc.
        - dwellTime_dict: Dictionary containing distributions of dwell times for different scenarios.
        - assignDwellTime: Function that assigns dwell times to trips based on the given distribution.
        - print_progress: Boolean indicating whether to display progress information during execution.

        Returns:
        - DataFrame with adjusted dwell times and updated trip times to ensure daily activities fit within a 24-hour period.
        """
        
        # Create a copy of the DataFrame to avoid modifying the original data in place.
        updated_df = df.copy()

        # Filter out rows where the end time exceeds 24 hours (1440 minutes) to identify trips that need adjustment.
        filtered_df = df[df['end_min'] > 1440]

        # Extract unique pairs of 'uniqID' and 'Day_Type' that meet the filter condition for further processing.
        unique_pairs = filtered_df[['uniqID', 'Day_Type']].drop_duplicates()

        # Process each unique pair to adjust the dwell times and ensure the total schedule fits within 24 hours.
        iterator = tqdm(unique_pairs.iterrows(), total = len(unique_pairs), desc = '6. Time constraint') if print_progress else unique_pairs.iterrows()
        
        for _, row in iterator:
            # Extract trips for the current unique pair.
            group_df = df[(df['uniqID'] == row['uniqID']) & (df['Day_Type'] == row['Day_Type'])]

            # Attempt to adjust the dwell times up to 100 times to fit the schedule within 24 hours.
            for attempt in range(1, 101):
                # Assign dwell times using the provided function, which may incorporate randomness or specific logic.
                updated_group_df = assignDwellTime(group_df, dwellTime_dict, print_ = False)
                # Recalculate start and end times based on the newly assigned dwell times.
                updated_group_df = update_rows(updated_group_df)

                # Check if the adjustments have successfully brought the schedule within 24 hours.
                if updated_group_df['end_min'].iloc[-1] <= 1440:
                    break

            # If the schedule still exceeds 24 hours after 100 attempts, issue a warning.
            if updated_group_df['end_min'].iloc[-1] > 1440:
                print(f"Warning: After 100 attempts, 'end_min' is still above 2400 for uniqID {row['uniqID']} and Day_Type {row['Day_Type']}")

            if 'trip_count_class' in updated_group_df.columns:
                updated_group_df = updated_group_df.drop('trip_count_class', axis=1) 

            # Update the main DataFrame with the adjusted trip times for the current unique pair.
            updated_df.update(updated_group_df)

            updated_df['uniqID'] = updated_df['uniqID'].astype('int64')
            updated_df['Home_cbg'] = updated_df['Home_cbg'].astype('int64')
            updated_df['sequence'] = updated_df['sequence'].astype('int64')

        return updated_df


    #-----------------------
    #-----------------------
    #-----------------------
    updated_df = timeConstraint(updated_df, dwellTime_dict, assignDwellTime, print_progress)
    #-----------------------
    #-----------------------
    #-----------------------

    # If there are duplicate trips printed, adjust dwell time
    def logicalConstraint2(df, print_progress):
        """
        Adjusts simulated trips by identifying and merging consecutive trips that have the same
        destination and purpose, effectively simulating a more realistic scenario where such trips
        are part of a single, longer stay at the destination rather than multiple, separate trips.

        Parameters:
        - df: DataFrame containing the simulated trip data, including 'Dest' and 'TRPPURP' columns.
        - print_progress: Boolean indicating whether to show progress during the execution.

        Returns:
        - A DataFrame with consecutive trips to the same destination and for the same purpose merged,
          resulting in adjusted 'Dwell_Time' and start/end times for the trips.
        """
        
        def process_group(group):
            """
            Identifies consecutive trips within a group (for the same individual on the same day) that should be merged
            based on having the same destination and purpose. Adjusts the 'Dwell_Time' and start/end times for merged trips.

            Parameters:
            - group: DataFrame segment representing all trips for an individual on a specific day.

            Returns:
            - The group with adjusted trips and a list of row indices that should be removed because their trips have been merged.
            """
            
            # Create a flag indicating where the 'TRPPURP' or 'Dest' changes from the previous row.
            trppurp_changed = group['TRPPURP'] != group['TRPPURP'].shift()
            dest_changed = group['Dest'] != group['Dest'].shift()
            boundaries = (trppurp_changed | dest_changed)

            # Where either flag is True, a new 'group_id' is started to identify trips that can be merged.
            group['group_id'] = boundaries.cumsum()

            # Identify first and the last raw of consecutive groups
            first_rows = group.drop_duplicates(subset=['group_id'], keep='first')
            last_rows = group.drop_duplicates(subset=['group_id'], keep='last')

            # Calculate end_min and Dwell_Time for the last raw
            group.loc[first_rows.index, 'end_min'] = last_rows['end_min'].values
            group.loc[first_rows.index, 'Dwell_Time'] = group.loc[first_rows.index, 'end_min'] - group.loc[first_rows.index, 'start_min']

            # end_min = 1440
            group.iloc[-1, group.columns.get_loc('end_min')] = 1440
            group.iloc[-1, group.columns.get_loc('Dwell_Time')] = 1440 - group.iloc[-1, group.columns.get_loc('start_min')]

            # filter duplicated raws based on group_id
            all_duplicated = group.duplicated(subset=['group_id'], keep=False)
            
            # extract index of duplicated raws except first raws
            index_to_remove = group[all_duplicated & (~group.index.isin(first_rows.index))].index.tolist()
    #         print(index_to_remove)

            # Delete column group_id
            group = group.drop(columns=['group_id'])

            return group, index_to_remove

        # Create a copy of the DataFrame to avoid modifying the original data in place.
        df_copy = df.copy()
        # List to store indices of rows that need to be removed after merging consecutive trips.
        indices_to_remove = []

        # Process each group of trips by the same individual on the same day.
        iterator = tqdm(df.groupby(['uniqID', 'Day_Type']), desc = '7. Logical constraint (duplicate trips)') if print_progress else df.groupby(['uniqID', 'Day_Type'])
        
        for _, group in iterator:
            processed_group, to_remove = process_group(group)
            # Update the processed group back into the DataFrame copy.
            df_copy.loc[processed_group.index] = processed_group
            
            # Append the indices of rows to be removed to the list.
            indices_to_remove.extend(to_remove)

        df_copy = df_copy.drop(indices_to_remove)
        # After merging trips, the sequence of trips for each day might be disrupted.
        # Recalculate the sequence to ensure it's continuous and starts from 1 for each day.
        df_copy['sequence'] = df_copy.groupby(['uniqID', 'Day_Type']).cumcount() + 1
        # Reset the DataFrame index for cleanliness and to reflect the removal of certain rows.
        df_copy.reset_index(drop=True, inplace=True)
        return df_copy


    #-----------------------
    #-----------------------
    #-----------------------
    updated_df = logicalConstraint2(updated_df, print_progress)
    #-----------------------
    #-----------------------
    #-----------------------
    
    return updated_df


# # Code for data preprocessing
# ## A1. Preprocess NHTS data
# ### A1.1. Organize columns of NHTS data

# In[15]:


def organize_columns(df, print_progress = True):
    """
    This function reorganizes the NHTS (National Household Travel Survey) dataset by selecting important columns for analysis, 
    mapping certain categorical codes to more meaningful values, and introducing new columns to better represent the data. 
    It focuses on clarifying trip purposes, transportation modes, and travel days based on NHTS coding schemes. 
    The function aims to make the dataset more accessible and informative for subsequent analysis.

    - HOUSEID and PERSONID are maintained as identifiers.
    - Trip purposes (TRPPURP) are derived from WHYTO, translating numerical codes to readable categories 
      like 'Home', 'Work', etc.
    - Transportation modes (TRPTRANS) are mapped to categories like 'Walk', 'Bicy', 'Car', 'PTrans' 
      (public transport), and 'Air', based on the TRPTRANS codes.
    - Travel days (TRAVDAY) are categorized into 'Weekend' or 'Weekday' to facilitate analysis based 
      on the day of the week.
    - Respondent ages (R_AGE, R_AGE_IMP) are reclassified into broader age groups for more general analysis.

    Parameters:
    - df: The DataFrame containing NHTS trip data to be organized.
    - print_progress: A Boolean flag that indicates whether progress should be printed during execution.

    Returns:
    - A DataFrame (trippub_re) that has been reorganized and cleaned for easier analysis, with selected columns 
      and new mappings applied to enhance readability and interpretability.
    """
    
    trippub_re = df.copy()
    
    # Explicitly select and reorder columns relevant for analysis from the NHTS data.
    columns = ['HOUSEID', 'PERSONID', 'HHSTFIPS','R_AGE', 'R_AGE_IMP', 'TRIPPURP', 'WHYTRP1S', 'WHYTRP90', 'WHYFROM', 'WHYTO', 'TRPMILES', 'DWELTIME','STRTTIME', 'ENDTIME', 'TRPTRANS','TRAVDAY', 'TDAYDATE']
    trippub_re = trippub_re[columns]
    
    # 0) TRPPURP_column
    tqdm.pandas(desc="0) mapping TRPPURP")
    
    # Map the 'WHYTO' column to new, more understandable trip purpose labels.
    def map_purpose(row):
        if row['WHYTO'] in [1, 2]:
            return 'Home'
        elif row['WHYTO'] in [3, 4]:
            return 'Work'
        elif row['WHYTO'] in [6, 8, 9, 10, 19]:
            return 'S_d_r'
        elif row['WHYTO'] == 11:
            return 'D_shop'
        elif row['WHYTO'] == 13:
            return 'Meals'
        elif row['WHYTO'] == 17:
            return 'V_fr_rel'
        elif row['WHYTO'] in [15, 16]:
            return 'Rec_lei'
        elif row['WHYTO'] in [12, 14, 18]:
            return 'Serv_trip'
        elif row['WHYTO'] in [5, 97]:
            return 'Others'
        else:
            return None
        
        
    if print_progress == True:
        trippub_re['TRPPURP_new'] = trippub_re.progress_apply(map_purpose, axis=1)
    else:
        trippub_re['TRPPURP_new'] = trippub_re.apply(map_purpose, axis=1)
    

    # Map transportation modes from the 'TRPTRANS' column to more readable labels.
    tqdm.pandas(desc="1) mapping TRPTRANS")
    
    def map_mode(row):
        if row['TRPTRANS'] == 1:
            return 'Walk'
        elif row['TRPTRANS'] == 2:
            return 'Bicy'
        elif row['TRPTRANS'] in [3,4,5,6,8,9,10,18]:
            return 'Car'
        elif row['TRPTRANS'] in [10,11,12,13,14,16]:
            return 'PTrans'
        elif row['TRPTRANS'] == 19:
            return 'Air'
        else:
            return None
    
    if print_progress == True:    
        trippub_re['TRPTRANS_new'] = trippub_re.progress_apply(map_mode, axis=1)
    else:
        trippub_re['TRPTRANS_new'] = trippub_re.apply(map_mode, axis=1)
    
    
    
    # Categorize days of travel into weekdays and weekends based on 'TRAVDAY'.
    def map_week(row):
        if row['TRAVDAY'] in [1,7]:
            return 'Weekend'
        elif row['TRAVDAY'] in [2,3,4,5,6]:
            return 'Weekday'
        else:
            return None
    
    
    if print_progress == True:    
        trippub_re['TRAVDAY_new'] = trippub_re.progress_apply(map_week, axis=1)
    else:
        trippub_re['TRAVDAY_new'] = trippub_re.apply(map_week, axis=1)
    
    
    # Reclassify 'R_AGE_IMP' into broader age groups to facilitate demographic analysis.
    tqdm.pandas(desc="3) mapping R_AGE_new")
    
    def reclassify_age(age):
        if age < 10:
            return 'Child'
        elif 10 <= age < 20:
            return 'Teen'
        elif 20 <= age < 40:
            return 'Adult'
        elif 40 <= age < 60:
            return 'MidAdult'
        else:
            return 'Seniors'

    if print_progress == True:    
        trippub_re['R_AGE_new'] = trippub_re['R_AGE_IMP'].progress_apply(reclassify_age)
    else:
        trippub_re['R_AGE_new'] = trippub_re['R_AGE_IMP'].apply(reclassify_age)
    
    
    if print_progress == True:   
        print()
        print('4) Now, Other columns...')
    
    # Merge 'HOUSEID' and 'PERSONID' into a new identifier for unique respondents.
    trippub_re['HOUSEID'] = trippub_re['HOUSEID'].astype(str)
    trippub_re['PERSONID'] = trippub_re['PERSONID'].astype(str)
    trippub_re['PERSONID_new'] = trippub_re['HOUSEID'] + trippub_re['PERSONID']
    
    num_empty_rows = trippub_re['TRPPURP_new'].isnull().sum()

    # Remove rows with missing new trip purpose to ensure dataset completeness.
    trippub_re.dropna(subset=['TRPPURP_new'], inplace=True)
    trippub_re.dropna(subset=['TRPTRANS_new'], inplace=True)
    trippub_re.dropna(subset=['R_AGE_IMP'], inplace=True)
    trippub_re.dropna(subset=['TRAVDAY_new'], inplace=True)
    
    
    if print_progress == True:   
        print()
        print('... done!')
        
    # Filtering Columns
    selected_columns = ['HOUSEID', 'PERSONID', 'PERSONID_new', 'HHSTFIPS', 'R_AGE_IMP', 'R_AGE_new', 'WHYFROM', 'WHYTO', 'TRPMILES', 'DWELTIME',
                        'STRTTIME', 'ENDTIME', 'TRAVDAY', 'TRAVDAY_new', 'TDAYDATE', 'TRPPURP_new', 'TRPTRANS_new']
    
    trippub_re = trippub_re[selected_columns]
    
    
    return trippub_re


# ### A1.2. Preprocess Data

# In[16]:


def preprocess_NHTS(df, print_progress = True):
    """
    Cleans and preprocesses the NHTS dataset (After running organize_columns function) to correct data anomalies and enhance data quality for analysis. 
    The preprocessing steps include adjusting dwelling times, calculating travel times, and ensuring data consistency across the travel records.

    Parameters:
    - df: The filtered NHTS DataFrame (returned from organize_columns function) to preprocess.
    - print_progress: If true, prints the progress of preprocessing steps.

    Returns:
    - A preprocessed DataFrame with corrected and calculated fields relevant for travel analysis.
    """
    
    trip_total = df.copy()
    
    if print_progress == True:
        print('Start preprocessing... NHTS data...')
      
    # Step 1 ~ 2
    # Correct dwelling times: Set dwelling times less than 0 to 0.
    trip_total['DWELTIME'] = trip_total['DWELTIME'].apply(lambda x: 0 if x < 0 else x)
    
    # Extract hours and minutes from ENDTIME and STRTTIME to calculate travel times.
    trip_total['ARRIVAL_hour'] = trip_total['ENDTIME'] // 100
    trip_total['ARRIVAL_minute'] = trip_total['ENDTIME'] % 100
    trip_total['STRTTIME_hour'] = trip_total['STRTTIME'] // 100
    trip_total['STRTTIME_minute'] = trip_total['STRTTIME'] % 100
    
    # Convert arrival and start times to minutes.
    trip_total['ARRIVAL_in_minutes'] = trip_total['ARRIVAL_hour'] * 60 + trip_total['ARRIVAL_minute']
    trip_total['STRTTIME_in_minutes'] = trip_total['STRTTIME_hour'] * 60 + trip_total['STRTTIME_minute']
    
    # Calculate the end time in minutes by adding dwelling time to arrival time.
    trip_total['ENDTIME_minute'] = trip_total['ARRIVAL_in_minutes'] + trip_total['DWELTIME']
    
    # Calculate travel time in minutes as the difference between arrival and start times.
    trip_total['TRAVTIME'] = trip_total['ARRIVAL_in_minutes'] - trip_total['STRTTIME_in_minutes']

    # Adjust for cases where travel time calculation rolls over a day boundary.
    trip_total['TRAVTIME'] = trip_total['TRAVTIME'].apply(lambda x: x if x >= 0 else x + 2400)
    
   
    # Remove records where the sum of dwelling time for a unique person ID and travel day combination is 0.
    sum_dweltime = trip_total.groupby(['PERSONID_new', 'TRAVDAY_new'])['DWELTIME'].sum().reset_index()
    valid_ids = sum_dweltime[sum_dweltime['DWELTIME'] != 0][['PERSONID_new', 'TRAVDAY_new']]
    trip_total = pd.merge(trip_total, valid_ids, on=['PERSONID_new', 'TRAVDAY_new'])
    
    # Rename columns for clarity and consistency with analysis needs.
    trip_total = trip_total.rename(columns={
        'PERSONID_new': 'uniqID',
        'R_AGE_IMP': 'age_Group',
        'TRAVDAY_new': 'Day_Type',
        'TRPPURP_new': 'Trip_pur',
        'DWELTIME': 'Dwell_T_min',
        'TRAVTIME': 'Trip_T_min',
        'STRTTIME': 'sta_T_hms',
        'STRTTIME_in_minutes': 'sta_T_min',
        'ENDTIME': 'arr_T_hms',
        'ARRIVAL_in_minutes': 'arr_T_min',
        'ENDTIME_minute' : 'end_T_min'
    })
    
    # Organize the preprocessed data for further steps.
    trip_total = trip_total[['uniqID', 'age_Group', 'Day_Type', 'Trip_pur', 'sta_T_hms', 'arr_T_hms', 'Dwell_T_min', 'Trip_T_min', 'sta_T_min', 'arr_T_min', 'end_T_min']]
    trip_total = trip_total.sort_values(['Day_Type', 'uniqID', 'sta_T_min'])

    tqdm.pandas(desc="0) remove duplicate home")
    
    # Step 3
    def remove_duplicate_home(group):
        if len(group) > 1:
            if group.iloc[0]['Trip_pur'] == 'Home' and group.iloc[1]['Trip_pur'] == 'Home':
                return group.iloc[1:]
        return group

    trip_total = trip_total.sort_values(['Day_Type', 'uniqID', 'sta_T_min'])

    # Remove duplicated Home
    if print_progress == True:
        trip_total = trip_total.groupby(['Day_Type', 'uniqID']).progress_apply(remove_duplicate_home).reset_index(drop=True)
    else:
        trip_total = trip_total.groupby(['Day_Type', 'uniqID']).apply(remove_duplicate_home).reset_index(drop=True)

    # display(trip_total[trip_total['uniqID'] ==300010541])

    # Step 4
    tqdm.pandas(desc="1) Start trip from home")

    def add_home_if_needed(group):
        
        if group.iloc[0]['sta_T_min'] == 0:
            return None
    
        if group.iloc[0]['Trip_pur'] != 'Home':
            new_row = group.iloc[0].copy()
            new_row['Trip_pur'] = 'Home'
            new_row['Dwell_T_min'] = group.iloc[0]['sta_T_min']
            new_row['Trip_T_min'] = 0
            new_row['sta_T_min'] = 0
            new_row['arr_T_min'] = 0
            new_row['end_T_min'] = new_row['Dwell_T_min']
            group = pd.concat([pd.DataFrame([new_row]), group]).reset_index(drop=True)
        return group
    
    if print_progress == True:
        trip_total = trip_total.groupby(['Day_Type', 'uniqID']).progress_apply(add_home_if_needed).reset_index(drop=True).dropna()
    else:
        trip_total = trip_total.groupby(['Day_Type', 'uniqID']).apply(add_home_if_needed).reset_index(drop=True).dropna()
    
    
    tqdm.pandas(desc="2) Adjusting initial Home Time")

    # Step 5: adjust_home_time function corrects the start and end times for the first 'Home' trip of the day
    def adjust_home_time(group):
        first_row = group.iloc[0]
        if first_row['Trip_pur'] == 'Home':
            # print(first_row['uniqID'])

            # display(group)
            next_row_idx = first_row.name + 1
            # print(next_row_idx)
            if next_row_idx in group.index:
                next_row = group.loc[next_row_idx]

                group.at[first_row.name, 'sta_T_min'] = 0
                group.at[first_row.name, 'arr_T_min'] = 0
                group.at[first_row.name, 'end_T_min'] = next_row['sta_T_min']
                group.at[first_row.name, 'Dwell_T_min'] = group.at[first_row.name, 'end_T_min']
                group.at[first_row.name, 'Trip_T_min'] = 0
            else:
                group.at[first_row.name, 'sta_T_min'] = 0
                group.at[first_row.name, 'arr_T_min'] = 0
                group.at[first_row.name, 'end_T_min'] = 1440
                group.at[first_row.name, 'Dwell_T_min'] = 1440
                group.at[first_row.name, 'Trip_T_min'] = 0
        return group

    
    if print_progress == True:
        trip_total = trip_total.groupby(['Day_Type', 'uniqID']).progress_apply(adjust_home_time).reset_index(drop=True)
    else:
        trip_total = trip_total.groupby(['Day_Type', 'uniqID']).apply(adjust_home_time).reset_index(drop=True)
    

    # Remove entries where the start time is not less than the end time, essentially removing records where travel times might extend beyond a 24-hour period, which could indicate data errors or unusual travel behavior.
    trip_total = trip_total[trip_total['sta_T_min'] < trip_total['end_T_min']]

    
    tqdm.pandas(desc="3) Addressing Dwell Time")  
    
    # Step 6: Filters out groups where dwell time is zero, as it's unrealistic for someone to have no dwell time at a location unless it's a pass-through, which wouldn't typically be recorded as a separate trip. This helps in maintaining the quality of the dataset by ensuring all recorded trips reflect actual stops.
    def filter_invalid_groups(group):
#         global deleted_rows_counter
        dwell_zero_count = sum(group['Dwell_T_min'] == 0)
        last_row = group.iloc[-1]

        # Delete groupes that disatisfy following conditions
        if dwell_zero_count >= 2:
#             deleted_rows_counter += len(group)
            return None
        if dwell_zero_count >= 1 and last_row['Trip_pur'] != 'Home':
#             deleted_rows_counter += len(group)
            return None
        if last_row['Trip_pur'] != 'Home' and last_row['Dwell_T_min'] == 0:
#             deleted_rows_counter += len(group)
            return None

        return group

    if print_progress == True:
        trip_total = trip_total.groupby(['Day_Type', 'uniqID']).progress_apply(filter_invalid_groups).reset_index(drop=True).dropna()
    else:
        trip_total = trip_total.groupby(['Day_Type', 'uniqID']).apply(filter_invalid_groups).reset_index(drop=True).dropna()


    # Step 7: Ensure each travel day ends at 'Home' for each unique ID and Day_Type combination.
    tqdm.pandas(desc="4) Add Home as a last travel")  # tqdm의 pandas 확장을 활성화
    
    def add_or_remove_home_at_end(group):

        # Check if the last trip purpose is not 'Home'.
        if group.iloc[-1]['Trip_pur'] != 'Home':
            last_row = group.iloc[-1]
            new_row = last_row.copy()
            new_row['Trip_pur'] = 'Home'
            new_row['sta_T_min'] = last_row['end_T_min']

            # Calculate the average travel time for non-zero travel times to estimate the arrival time.
            trip_t_min_mean = int(group[group['Trip_T_min'] != 0]['Trip_T_min'].mean())

            new_row['arr_T_min'] = trip_t_min_mean + new_row['sta_T_min']

            # If the estimated arrival time exceeds the daily limit, discard the group.
            if new_row['arr_T_min'] > 1440:
#                 deleted_rows_counter += len(group)
                return None

            new_row['end_T_min'] = 1440
            new_row['Trip_T_min'] = new_row['arr_T_min'] - new_row['sta_T_min']
            new_row['Dwell_T_min'] = 1440 - new_row['arr_T_min']

            # Update the start and arrival times to HHMM format.
            new_row['sta_T_hms'] = (new_row['sta_T_min'] // 60) * 100 + (new_row['sta_T_min'] % 60)
            new_row['arr_T_hms'] = (new_row['arr_T_min'] // 60) * 100 + (new_row['arr_T_min'] % 60)

            # Add the new 'Home' trip to the group.
            group = pd.concat([group, pd.DataFrame([new_row])]).reset_index(drop=True)
            # display(group)
        return group

    # Apply the function to each group and handle NaN values from the function (if any).
    if print_progress == True:
        trip_total = trip_total.groupby(['Day_Type', 'uniqID']).progress_apply(add_or_remove_home_at_end).reset_index(drop=True).dropna()
    else:
        trip_total = trip_total.groupby(['Day_Type', 'uniqID']).apply(add_or_remove_home_at_end).reset_index(drop=True).dropna()
    
    
    # Step 8: Remove consecutive 'Home' trips at the end of the day and adjust the columns accordingly.
    tqdm.pandas(desc="5) Remove duplicate last Home and Adjust")

    def remove_last_home(group):
        group.reset_index(inplace = True, drop=True)
        last_idx = len(group) - 1
        
        if last_idx > 0: # Ensure there are at least two trips to compare.
            if group.iloc[last_idx]['Trip_pur'] == 'Home' and group.iloc[last_idx - 1]['Trip_pur'] == 'Home':
                # print(group['uniqID'])
                group = group.iloc[:-1]  #Delete the last row
                
                # Adjust the remaining last trip's times to align with the day's end.
                group.reset_index(inplace = True, drop=True)
                new_last_idx = len(group) - 1
                group.at[new_last_idx, 'end_T_min'] = 1440
                group.at[new_last_idx, 'Dwell_T_min'] = 1440 - group.at[new_last_idx, 'arr_T_min']

                # Update time formats to HHMM for consistency.
                group.at[new_last_idx, 'sta_T_hms'] = (group.at[new_last_idx, 'sta_T_min'] // 60) * 100 + (group.at[new_last_idx, 'sta_T_min'] % 60)
                group.at[new_last_idx, 'arr_T_hms'] = (group.at[new_last_idx, 'arr_T_min'] // 60) * 100 + (group.at[new_last_idx, 'arr_T_min'] % 60)

                # display(group)
        return group
    
    # Apply the function to each group, ensuring that NaN values are handled correctly.
    if print_progress == True:
        trip_total = trip_total.groupby(['Day_Type', 'uniqID']).progress_apply(remove_last_home).reset_index(drop=True).dropna()
    else:
        trip_total = trip_total.groupby(['Day_Type', 'uniqID']).apply(remove_last_home).reset_index(drop=True).dropna()
     
    
    # Step 9: Adjust the time details for the final 'Home' trip of each day to ensure it accurately reflects the end of the day.
    tqdm.pandas(desc="6) Adjusting Last Home Time") 
    
    def adjust_home_time(group):
        group.reset_index(inplace = True, drop=True)
        
        last_idx = len(group) - 1
        if last_idx >= 0:  # Check there is at least one entry to adjust.
            if group.iloc[last_idx]['Trip_pur'] == 'Home':
                # Adjust the last 'Home' trip to ensure it represents the end of the day.
                group.at[last_idx, 'end_T_min'] = 1440
                
                group.at[last_idx, 'Dwell_T_min'] = 1440 - group.at[last_idx, 'arr_T_min']
                group.at[last_idx, 'sta_T_hms'] = (group.at[last_idx, 'sta_T_min'] // 60) * 100 + (group.at[last_idx, 'sta_T_min'] % 60)
                group.at[last_idx, 'arr_T_hms'] = (group.at[last_idx, 'arr_T_min'] // 60) * 100 + (group.at[last_idx, 'arr_T_min'] % 60)

        # display(group)
        return group
    
    # Apply the adjustment to each group.
    if print_progress == True:
        trip_total = trip_total.groupby(['Day_Type', 'uniqID']).progress_apply(adjust_home_time).reset_index(drop=True).dropna()
    else:
        trip_total = trip_total.groupby(['Day_Type', 'uniqID']).apply(adjust_home_time).reset_index(drop=True).dropna()

    
    # Step 10: Merge consecutive 'Home' trip records, excluding the first and last trip of each day, into a single 'Home' trip.
    tqdm.pandas(desc="7) merge consecutive home")  
    
    # Assuming repaired_orig_data is your DataFrame
    def merge_consecutive_home_groups(group):
        consecutive_home_groups = []
        current_group = []

        # Apply the consolidation function to each group.
        for idx, row in group.iterrows():
            if row['Trip_pur'] == 'Home':
                current_group.append(idx)
            else:
                if current_group:
                    consecutive_home_groups.append(current_group.copy())
                    current_group = []

        if current_group:
            consecutive_home_groups.append(current_group)

        # Merge consecutive home groups
        for home_group in consecutive_home_groups:
            if len(home_group) > 1:
                total_dwell_time = group.loc[home_group, 'Dwell_T_min'].sum()
                total_trip_time = group.loc[home_group, 'Trip_T_min'].sum()

                # Update the end time of the first row with the end time of the last row in the group
                group.at[home_group[0], 'end_T_min'] = group.at[home_group[-1], 'end_T_min']
            
                # Update the first row in the consecutive home group
                group.at[home_group[0], 'Dwell_T_min'] = group.at[home_group[0], 'end_T_min'] - group.at[home_group[0], 'arr_T_min']

                # Drop the rows in the consecutive home group except for the first one
                group = group.drop(home_group[1:])

        # Reset the index
        group = group.reset_index(drop=True)

        # Update sta_T_hms and arr_T_hms
        group['sta_T_hms'] = (group['sta_T_min'] // 60) * 100 + (group['sta_T_min'] % 60)
        group['arr_T_hms'] = (group['arr_T_min'] // 60) * 100 + (group['arr_T_min'] % 60)

        # display(group)
        return group

    # Apply the function to each group
    if print_progress == True:
        trip_total = trip_total.groupby(['Day_Type', 'uniqID']).progress_apply(merge_consecutive_home_groups).reset_index(drop=True).dropna()
    else:
        trip_total = trip_total.groupby(['Day_Type', 'uniqID']).apply(merge_consecutive_home_groups).reset_index(drop=True).dropna()

   
    # Step 11: Ensure that the end time of one trip matches the start time of the following trip.
    tqdm.pandas(desc="8) align 'start time' with next 'end time'")  # tqdm의 pandas 확장을 활성화
    
    def align_end_time_with_next_start_time(group):
        for idx in range(len(group) - 1):
            current_row = group.iloc[idx]
            next_row = group.iloc[idx + 1]

            if current_row['end_T_min'] != next_row['sta_T_min']:
                # Update end_T_min and Dwell_T_min of the current row
                group.at[current_row.name, 'end_T_min'] = next_row['sta_T_min']
                group.at[current_row.name, 'Dwell_T_min'] = next_row['sta_T_min'] - current_row['arr_T_min']

        return group

    # Apply the function to each group
    if print_progress == True:
        trip_total = trip_total.groupby(['Day_Type', 'uniqID']).progress_apply(align_end_time_with_next_start_time).reset_index(drop=True)
    else:
        trip_total = trip_total.groupby(['Day_Type', 'uniqID']).apply(align_end_time_with_next_start_time).reset_index(drop=True)
    
    
    # Step 12: Remove groups that do not start or end with a 'Home' trip or where times do not make logical sense.
    tqdm.pandas(desc="9) filter Home as a start and end pur")  
    
    def filter_group(group):
#         global deleted_rows_counter
        
        group.reset_index(inplace=True, drop=True)
        last_idx = len(group) - 1
        first_idx = 0

        if last_idx >= 0:  # inspection only if there is at least one row in the group
            first_condition = group.at[first_idx, 'sta_T_min'] != 0
            last_condition = group.at[last_idx, 'end_T_min'] != 1440
            trip_pur_condition = group.at[last_idx, 'Trip_pur'] != 'Home'
            
            if first_condition or last_condition or trip_pur_condition:
#                 deleted_rows_counter += len(group)
                return None  # This group does not logically conclude or initiate with 'Home'.

        return group

    # Apply the function to each group and filter out None values
    if print_progress == True:
        trip_total = trip_total.groupby(['Day_Type', 'uniqID']).progress_apply(filter_group).reset_index(drop=True).dropna()
    else:
        trip_total = trip_total.groupby(['Day_Type', 'uniqID']).apply(filter_group).reset_index(drop=True).dropna()
    
    
    # Step 13: Refine start, arrival, and end times to HHMM format based on minute values.
    tqdm.pandas(desc="10) refine start, arr, end time with hms format")  # tqdm의 pandas 확장을 활성화
        
    def hms_format(group):        
        group['sta_T_hms'] = (group['sta_T_min'] // 60) * 100 + (group['sta_T_min'] % 60)
        group['arr_T_hms'] = (group['arr_T_min'] // 60) * 100 + (group['arr_T_min'] % 60)
        
        # Add end_T_min
        group['end_T_hms'] = (group['end_T_min'] // 60) * 100 + (group['end_T_min'] % 60)
        
        return group
        
    
    # Apply the function to each group and filter out None values
    trip_total['end_T_hms'] = None
    
    if print_progress == True:
        trip_total = trip_total.groupby(['Day_Type', 'uniqID']).progress_apply(hms_format).reset_index(drop=True).dropna()
    else:
        trip_total = trip_total.groupby(['Day_Type', 'uniqID']).apply(hms_format).reset_index(drop=True).dropna()


    # Step 14: Reclassify ages into categorical groups for easier analysis.
    tqdm.pandas(desc="11) Reclassifying age variables")  
    
    def reclassify_age(age):
        if age < 10:
            return 'Child'
        elif 10 <= age < 20:
            return 'Teen'
        elif 20 <= age < 40:
            return 'Adult'
        elif 40 <= age < 60:
            return 'MidAdult'
        else:
            return 'Seniors'

        
    if print_progress == True:
        trip_total['age_class'] = trip_total['age_Group'].progress_apply(reclassify_age)
    else:
        trip_total['age_class'] = trip_total['age_Group'].apply(reclassify_age)
    
    if "Unnamed: 0" in trip_total.columns:
        trip_total.drop("Unnamed: 0", axis=1, inplace=True)
    
    trip_total.rename(columns={"age_Group": "age"}, inplace=True)
    
        
    return trip_total[['uniqID', 'age', 'age_class', 'Day_Type', 'Trip_pur', 'sta_T_hms', 'arr_T_hms', 'end_T_hms', 'Dwell_T_min', 'Trip_T_min', 'sta_T_min', 'arr_T_min', 'end_T_min']]


# ### A1.3. Preprocessing for tripmode

# In[17]:


# 새로운 컬럼 TRPPURP_new 생성
def preprocess_NHTS_tripMode(df, print_progress = True):
    """
    Processes the NHTS dataset to create a new trip purpose category and calculates the probability of trip mode choices by age and newly defined trip purpose. This function aims to simplify the analysis of trip behaviors across different demographics and trip purposes by mapping detailed purposes into broader categories and excluding air travel from the analysis.

    Parameters:
    - df : DataFrame
        The NHTS DataFrame after initial processing, which includes WHYFROM codes that need to be mapped to broader trip purposes.
    - print_progress : bool, optional
        A flag to print progress updates during the execution of the function, by default True.

    Steps:
    1. Maps the WHYFROM column values to a new set of trip purpose categories.
    2. Removes any unnecessary columns, specifically 'Unnamed: 0' if present.
    3. Renames columns for clarity.
    4. Drops duplicate rows based on a subset of columns to ensure unique records for aggregation.
    5. Filters out trips made by air as they are not relevant for the probability calculation.
    6. Aggregates the data to count each trip mode within each age class and trip purpose combination.
    7. Counts the total number of records for each age class and trip purpose to serve as the denominator in the probability calculation.
    8. Merges the counts and total counts dataframes to calculate the probability of choosing each trip mode for the given demographics and trip purposes.

    Returns:
    - result_total : DataFrame
        A DataFrame containing the age class, trip purpose, trip mode, and the calculated probability of selecting each trip mode for each age class and trip purpose combination.
    """
    
    trippub_total = df.copy()
    
    tqdm.pandas(desc="0) Mapping trippurpose") 
    
    # Maps WHYFROM values to more generalized trip purpose categories.
    def map_purpose(row):
        if row['WHYFROM'] in [1, 2]:
            return 'Home'
        elif row['WHYFROM'] in [3, 4]:
            return 'Work'
        elif row['WHYFROM'] in [6, 8, 9, 10, 19]:
            return 'S_d_r'
        elif row['WHYFROM'] == 11:
            return 'D_shop'
        elif row['WHYFROM'] == 13:
            return 'Meals'
        elif row['WHYFROM'] == 17:
            return 'V_fr_rel'
        elif row['WHYFROM'] in [15, 16]:
            return 'Rec_lei'
        elif row['WHYFROM'] in [12, 14, 18]:
            return 'Serv_trip'
        elif row['WHYFROM'] in [5, 97]:
            return 'Others'
        else:
            return None
        
    # Apply the mapping function to create a new column for the mapped trip purposes.
    trippub_total['TRPFROM_new'] = trippub_total.progress_apply(map_purpose, axis=1)
    
    # Removes an unnecessary column if present.
    if "Unnamed: 0" in trippub_total.columns:
        trippub_total.drop('Unnamed: 0', axis=1, inplace=True)
    
    # replace name of R_AGE_new with age_class
    trippub_total.rename(columns={'R_AGE_new': 'age_class'}, inplace=True)
    
    if print_progress == True:
        print('Compute probability of trip mode choice by age and trip purpose ... ')
    
    # Drops duplicate rows based on specific columns to ensure unique records for the aggregation.
    trippub_total = trippub_total.drop_duplicates(subset=['PERSONID_new', 'age_class', 'TRPPURP_new', 'TRPTRANS_new'])
    trippub_total = trippub_total[['PERSONID_new', 'age_class', 'TRPPURP_new', 'TRPTRANS_new']]

    trippub_total.rename(columns={'PERSONID_new': 'uniqID'}, inplace=True)
    trippub_total.rename(columns={'TRPPURP_new': 'Trip_pur'}, inplace=True)
    trippub_total.rename(columns={'TRPTRANS_new': 'Trip_mode'}, inplace=True)
    trippub_total.reset_index(inplace = True, drop = True)
    
    # Count the number of each Trip_mode per age_class and Trip_pur
    trippub_total = trippub_total[trippub_total['Trip_mode'] != 'Air'] 

    count_df_total = trippub_total.groupby(['age_class', 'Trip_pur', 'Trip_mode']).size().reset_index(name='nominator')

    # Count the total number of records per age_class and Trip_pur
    total_df_total = trippub_total.groupby(['age_class', 'Trip_pur']).size().reset_index(name='denominator')

    # Merge the two dataframes on age_class and Trip_pur
    result_total = pd.merge(count_df_total, total_df_total, on=['age_class', 'Trip_pur'])

    # Calculate the probability
    result_total['Trip_modeP'] = result_total['nominator'] / result_total['denominator']
    result_total
    
    print('Done!')
    
    return result_total


# ## A2. Preprocess Safegraph Data to Compute probability of trips from origin cbg to dest cbg
# ### A2.1. DO to OD, Covert Destination area to Origin area

# In[18]:


def DOtoOD(df, print_progress = True):
    """
    Converts Destination-Origin (DO) data into Origin-Destination (OD) data using the input SafeGraph Neighborhood data.
    The function processes columns related to work behavior and device home areas during weekdays and weekends,
    transforming them into a format that specifies how many devices move from one area to another.
    
    The process involves three main steps
    1) Calculating work behavior flows: Aggregates device counts moving from their home areas to work areas.
    2) Calculating weekday behavior flows: Similar aggregation for devices traveling based on weekday patterns.
    3) Calculating weekend behavior flows: Aggregates travels for devices during the weekend.
        
    Parameters:
    - df (pd.DataFrame): The DataFrame containing the DO data to be converted.
    - print_progress (bool): If True, prints progress updates during the calculation process.
    
    Returns:
    - pd.DataFrame: A DataFrame with the calculated OD data, including columns for area,
      work behavior from the area, weekday device home from the area, and weekend device home from the area.
    """
    
    ODdata = df.copy()

    def calculate_work_behavior_from(df):
        # Initialize a dictionary to store visits for each area, using area values from the DataFrame as keys.
        all_area_dict = {str(area): {} for area in df['area'].tolist()}

        # Iterate over each row of the DataFrame to calculate work behavior flows from one area to another.
        for _, row in tqdm(df.iterrows(), total = len(df), desc = '1. Work behavior...'):
            # Parse the 'work_behavior_device_home_areas' column as a dictionary to get current area dict.
            current_area_dict = eval(row['work_behavior_device_home_areas'])
            destination = str(row['area'])

            for source, count in current_area_dict.items():
                # Skip the iteration if the source area is not in the all_area_dict.
                if source not in all_area_dict:
                    continue

                # Initialize the destination in the source's dict if it doesn't exist.
                if destination not in all_area_dict[source]:
                    all_area_dict[source][destination] = 0

                # Add the count of devices moving from source to destination.
                all_area_dict[source][destination] += count

        # Map the all_area_dict data to a new column 'work_behavior_from_area' in the DataFrame.
        df['work_behavior_from_area'] = df['area'].map(lambda x: all_area_dict[str(x)])

        return df

    def fix_malformed_dict_str(s):
        # If the string does not end with a closing brace, attempt to correct it.
        if not s.endswith("}"):
            # Find the index of the last comma, which is where the string was likely cut off.
            last_comma_index = s.rfind(",")
            # Correct the string by ending it at the last comma and adding a closing brace.
            s = s[:last_comma_index] + "}"
        return s

    def calculate_weekday_behavior_from(df):
        # Initialize a dictionary to store visits for each area.
        all_area_dict = {str(area): {} for area in df['area'].tolist()}

        # Iterate over each row in the DataFrame.
        for _, row in tqdm(df.iterrows(), total = len(df), desc = '2) Other behaviors - weekday...'):
            # Correct any malformed dictionary strings.
            fixed_str = fix_malformed_dict_str(row['weekday_device_home_areas'])
            try:
                # Attempt to parse the corrected string into a dictionary.
                current_area_dict = eval(fixed_str)
            except Exception as e:
                # If parsing fails, print an error message and skip this row.
                print(f"Error in row {_}: {e}")
                continue

            destination = str(row['area'])

            for source, count in current_area_dict.items():
                # If the source area is not recognized, skip it.
                if source not in all_area_dict:
                    continue

                # Initialize the destination area in the source's dictionary if necessary.
                if destination not in all_area_dict[source]:
                    all_area_dict[source][destination] = 0

                # Increment the count of devices moving from source to destination.
                all_area_dict[source][destination] += count

        # Assign the calculated data to a new column in the DataFrame.
        df['weekday_device_from_area_home'] = df['area'].map(lambda x: all_area_dict[str(x)])

        return df

    def calculate_weekend_behavior_from(df):
        # Initialize a dictionary to store visits for each area.
        all_area_dict = {str(area): {} for area in df['area'].tolist()}

        for _, row in tqdm(df.iterrows(), total = len(df), desc = '3) Other behaviors - weekend...'):
            fixed_str = fix_malformed_dict_str(row['weekend_device_home_areas'])
            try:
                current_area_dict = eval(fixed_str)
            except Exception as e:
                print(f"Error in row {_}: {e}")
                continue

            destination = str(row['area'])

            for source, count in current_area_dict.items():
                if source not in all_area_dict:
                    continue

                if destination not in all_area_dict[source]:
                    all_area_dict[source][destination] = 0

                all_area_dict[source][destination] += count

        df['weekend_device_from_area_home'] = df['area'].map(lambda x: all_area_dict[str(x)])

        return df
    
    # 1) Work behavior
    ODdata = calculate_work_behavior_from(ODdata)
    
    # 2) Other behaviors - Weekday
    ODdata = calculate_weekday_behavior_from(ODdata)
    
    # 3) Other behaviors - Weekend
    ODdata = calculate_weekend_behavior_from(ODdata)

    # Extracting columns
    col = ['area', 'work_behavior_from_area','weekday_device_from_area_home','weekend_device_from_area_home']
    ODdata = ODdata[col]
    
    return ODdata


# ### A2.2. Computing probability of trips from origin cbg to dest cbg

# In[19]:


def compute_probabilityByk_Ws_Wd(neighbor_safegraphDF, landuseGDF, W_s, W_d):
    """
    Computes the probability of moving from an origin CBG to a destination CBG,
    factoring in the spatial attractiveness weight (W_s) and the distance sensitivity index (W_d).
    
    Parameters:
    - neighbor_safegraphDF (DataFrame): The DataFrame containing origin-destination data derived from the previous step (DOtoOD function).
    - landuseGDF (GeoDataFrame): A GeoDataFrame containing land use data for different CBGs.
    - W_s (float): The spatial attractiveness weight, affecting preference for destinations based on trip purpose density.
    - W_d (float): The distance sensitivity index, affecting preference for closer destinations.
    
    Returns:
    - DataFrame: A DataFrame with probabilities adjusted by W_s and W_d for trips from each origin to possible destinations.
    """

    def compute_probability(df, area='area', cols=['work_behavior_from_area', 'weekday_device_from_area_home', 'weekend_device_from_area_home']):
        """
        Initializes and computes base probabilities of moving from one area to another based on work behavior, weekday, and weekend device home areas.
        The probabilities are normalized by the total count of movements (k) for each category.
        """
        # Create a copy of the original DataFrame for results
        prob_trips_in_space = df.copy()
        
        prob_trips_in_space['work_behavior_from_area'] = prob_trips_in_space['work_behavior_from_area'].astype(str)
        prob_trips_in_space['weekday_device_from_area_home'] = prob_trips_in_space['weekday_device_from_area_home'].astype(str)
        prob_trips_in_space['weekend_device_from_area_home'] = prob_trips_in_space['weekend_device_from_area_home'].astype(str)

        # Iterate over each row and normalize the movement counts to probabilities.
        for index, row in tqdm(prob_trips_in_space.iterrows(), total=prob_trips_in_space.shape[0], desc = '1) Probability A to A_i...1 (add k folmula)'):

            # Update each dictionary in the row based on its own total_k
            for col in cols:
                dict_data = eval(row[col]) # Convert the string back to a dictionary.
                # Convert the string back to a dictionary.
                total_k = sum(dict_data.values())  # Calculate the total k for the current column only

                # Convert counts to probabilities by dividing each count by the total.
                for key in dict_data:
                    if total_k != 0:  # Ensure not to divide by zero
                        dict_data[key] = dict_data[key] / total_k
                    else:
                        dict_data[key] = 0

                # Update the DataFrame with the normalized probabilities.
                prob_trips_in_space.at[index, col] = str(dict_data)  # Convert updated dictionary back to string representation

        # Duplicate 'work_behavior_from_area' into new columns for further processing.
        prob_trips_in_space['weekday_Work'] = prob_trips_in_space['work_behavior_from_area']
        prob_trips_in_space['weekend_Work'] = prob_trips_in_space['work_behavior_from_area']

        # Split 'weekday_device_from_area_home' into multiple columns for different trip purposes.
        weekday_cols = ['weekday_School', 'weekday_University', 'weekday_Dailycare', 'weekday_Religion', 'weekday_Large_shop', 'weekday_Etc_shop', 'weekday_Meals', 'weekday_V_fr_rel', 'weekday_Rec_lei', 'weekday_Serv_trip', 'weekday_Others']
        for col in weekday_cols:
            prob_trips_in_space[col] = prob_trips_in_space['weekday_device_from_area_home']

        # Repeat the process for weekend data.
        weekend_cols = [col.replace('weekday', 'weekend') for col in weekday_cols]
        for col in weekend_cols:
            prob_trips_in_space[col] = prob_trips_in_space['weekend_device_from_area_home']

        return prob_trips_in_space

    def apply_Ws_formula(prob_trips_in_space, landUse, W_s):
        df_ws = prob_trips_in_space.copy()
        exclude_columns = ['work_behavior_from_area', 'weekday_device_from_area_home', 'weekend_device_from_area_home']
        df_ws.drop(exclude_columns, axis=1, inplace=True)

        # Iterate over the rows and columns of df_ws
        for index, row in tqdm(df_ws.iterrows(), total=df_ws.shape[0], desc = '2) Probability A to A_i...2 (add Ws weight)'):
            for col in df_ws.columns:
                # Avoid processing non-dictionary columns and excluded columns
                if col == 'area':
                    continue

                # Extract the purpose from the column name
                purpose = "_".join(col.split('_')[1:])

                dict_data = eval(row[col])

                # Calculate C_Ai for each key in the dictionary
                C_Ai_dict = {key: landUse[landUse['CBGCODE'] == key]['TRPPURP'].apply(lambda x: purpose in x).sum() for key in dict_data.keys()}
    #             print(purpose)
    #             print(C_Ai_dict)

                # Calculate sum_j C_Aj
                sum_C_Aj = sum(C_Ai_dict.values())

                # Apply the formula
                new_prob_values = {}  # A new dictionary to store normalized probabilities
                for key, value in dict_data.items():
                    C_Ai = C_Ai_dict[key]
                    multiplier = (C_Ai / sum_C_Aj) ** W_s if sum_C_Aj != 0 else 0  # Ensure not to divide by zero
                    new_prob_values[key] = value * multiplier

                # Normalize the probabilities to sum up to 1
                total_probability = sum(new_prob_values.values())
                for key in new_prob_values:
                    new_prob_values[key] = new_prob_values[key] / total_probability if total_probability != 0 else 0

                df_ws.at[index, col] = str(new_prob_values)

        return df_ws
    
    def apply_Ws_formula_optimized(prob_trips_in_space, landUse, W_s):
        """
        Optimizes and adjusts the probabilities of moving from an origin to various destinations by considering the spatial attractiveness weight (W_s).
        This approach pre-calculates and indexes the attractiveness measures (density or concentration of facilities relevant to the trip purpose) for each destination, 
        streamlining the adjustment of probabilities.

        The spatial attractiveness weight (W_s) is used to adjust the base probabilities by the relative density of facilities that match the trip purpose at each destination. 
        This makes destinations with a higher concentration of relevant facilities more likely to be chosen, proportional to the value of W_s.
        """
        df_ws = prob_trips_in_space.copy()
        exclude_columns = ['work_behavior_from_area', 'weekday_device_from_area_home', 'weekend_device_from_area_home']
        df_ws.drop(exclude_columns, axis=1, inplace=True)

        # Pre-calculate C_Ai (attractiveness measure) for all destinations across all purposes.
        all_keys = set()
        for col in tqdm(df_ws.columns, desc = '2) Probability A to A_i...2 (add Ws weight)'):
            if col != 'area':
                df_ws[col].apply(lambda x: all_keys.update(eval(x).keys()))
        all_keys = list(all_keys)
        
        # Create a dictionary to index C_Ai values for each destination and purpose.
        C_Ai_dict = {key: {} for key in all_keys}
        for key in tqdm(all_keys, desc = ' --- 2.1) Indexing '):
            sub_df = landUse[landUse['CBGCODE'] == key]
            for col in df_ws.columns:
                if col != 'area':
                    purpose = "_".join(col.split('_')[1:])
                    C_Ai_dict[key][purpose] = sub_df['TRPPURP'].apply(lambda x: purpose in x).sum()

        # Adjust the base probabilities using the pre-calculated attractiveness measures.
        for index, row in tqdm(df_ws.iterrows(), total=df_ws.shape[0], desc=' --- 2.2) add Ws weight '):
            for col in df_ws.columns:
                if col == 'area':
                    continue

                # Extract the purpose from the column name
                purpose = "_".join(col.split('_')[1:])
                dict_data = eval(row[col])

                # Fetch pre-calculated attractiveness measures for the current purpose.
                local_C_Ai_values = [C_Ai_dict[key][purpose] for key in dict_data.keys()]
                sum_C_Aj = sum(local_C_Ai_values)

                # Apply the formula
                new_prob_values = {}
                for (key, value), C_Ai in zip(dict_data.items(), local_C_Ai_values):
                    multiplier = (C_Ai / sum_C_Aj) ** W_s if sum_C_Aj != 0 else 0
                    new_prob_values[key] = value * multiplier

                # Normalize the probabilities to sum up to 1
                total_probability = sum(new_prob_values.values())
                for key in new_prob_values:
                    new_prob_values[key] = new_prob_values[key] / total_probability if total_probability != 0 else 0

                df_ws.at[index, col] = str(new_prob_values)

        return df_ws

    
    
    def calculate_distance_meters(point1, point2):
        # WGS 84
        geod = Geod(ellps="WGS84")

        angle1,angle2,distance = geod.inv(point1.x, point1.y, point2.x, point2.y)

        return distance

    
    def apply_Wd_formula(df, landUse, W_d=0):
        """
        Adjusts the probabilities of moving from an origin to various destinations based on the distance sensitivity index (W_d). 
        This function calculates the distance between origin and destination CBGs and applies a decay function to adjust probabilities based on these distances.
        The closer a destination is to the origin, the higher the probability of movement towards it, adjusted by the W_d parameter.
        """
        
        df_wd = df.copy()
        
        # Iterate over the rows and columns of df_wd to adjust probabilities based on distance.
        for index, row in tqdm(df_wd.iterrows(), total=df_wd.shape[0], desc='3) Probability A to A_i...3 (add W_d weight)'):

            # Convert row['area'] to string for matching and get the geometry of the area.
            area_str = str(row['area'])
            area_geometry = landUse[landUse['CBGCODE'] == area_str].geometry.iloc[0]
            area_center = area_geometry.centroid

            for col in df_wd.columns:
                if col == 'area':
                    continue # Skip the 'area' column as it's not needed for probability adjustment.

                dict_data = eval(row[col]) # Convert the string representation back to a dictionary.
                keys_to_remove = [] # List to keep track of destinations with zero probability.

                for key, value in dict_data.items():
                    # Attempt to get the destination geometry; if missing, mark for removal.
                    destination_geometry_series = landUse[landUse['CBGCODE'] == key].geometry

                    if destination_geometry_series.empty:
#                         print(key)
                        dict_data[key] = 0# Mark the probability as zero if the destination geometry is missing.
                        continue

                    destination_geometry = destination_geometry_series.iloc[0]
                    destination_center = destination_geometry.centroid  # Calculate the centroid of the destination area.
                    distance = calculate_distance_meters(area_center, destination_center)
                    distance = distance/1000 # Convert distance to kilometers for the decay function.

                    # Adjust the probability based on distance using the W_d parameter.
                    dict_data[key] = value * np.exp(-W_d * distance)

                # Normalize the updated probabilities so they sum up to 1.
                total_probability = sum(dict_data.values())
                for key in dict_data:
                    dict_data[key] = round(dict_data[key] / total_probability if total_probability != 0 else 0, 5)

                for key in keys_to_remove:
                    del dict_data[key] # Remove destinations with zero probability.

                df_wd.at[index, col] = str(dict_data)  # Update the DataFrame with the adjusted probabilities.

        return df_wd
    
    def apply_Wd_formula_optimized(df, landUse, W_d=0):
        """
        Optimizes the adjustment of movement probabilities between origins and destinations based on the distance sensitivity index (W_d) by pre-computing geometric centroids for all areas. 
        This approach facilitates faster distance calculations and the efficient application of the distance decay function to the movement probabilities, 
        reflecting the preference for closer destinations.

        The function modifies the probabilities such that destinations closer to the origin are more likely to be selected, 
        with the degree of preference for proximity determined by the W_d parameter. 
        This is achieved by applying an exponential decay function based on the calculated distances between each origin and destination pair.
        """
        df_wd = df.copy()

        # Precompute centroids for all areas in the land use GeoDataFrame for faster lookup.
        landUse['centroid'] = landUse['geometry'].centroid

        # Create a dictionary for fast lookup of centroids
        centroid_lookup = landUse.set_index('CBGCODE')['centroid'].to_dict()

        rows_to_drop = []  ## Keep track of rows with undefined areas, if any.
        areas_to_drop = []

        # Iterate over the rows and columns of df_wd
        for index, row in tqdm(df_wd.iterrows(), total=df_wd.shape[0], desc='3) Probability A to A_i...3 (add W_d weight)'):

            area_str = str(row['area'])
            area_center = centroid_lookup.get(area_str)

            if area_center is None:  # If there's no centroid for the area, mark row for removal
                rows_to_drop.append(index)
                areas_to_drop.append(area_str)
                continue

            for col in df_wd.columns:
                if col == 'area':
                    continue

                dict_data = eval(row[col])
                keys_to_remove = []

                # Adjust each probability value based on the distance to the destination.
                for key, value in dict_data.items():
                    destination_center = centroid_lookup.get(key)

                    if destination_center is None:  # Check for missing destination geometry
                        dict_data[key] = 0
                    else:
                        distance = calculate_distance_meters(area_center, destination_center) / 1000
                        dict_data[key] = value * np.exp(-W_d * distance)

                    # If value is zero, mark for removal
                    if dict_data[key] == 0:
                        keys_to_remove.append(key)

                # Normalize the updated probabilities.
                total_probability = sum(dict_data.values())
                for key in dict_data:
                    dict_data[key] = round(dict_data[key] / total_probability if total_probability != 0 else 0, 5)

                # Remove destination probabilities set to zero.
                for key in keys_to_remove:
                    del dict_data[key]

                df_wd.at[index, col] = str(dict_data)

        # Drop rows where area_center is None and reset index
        df_wd.drop(rows_to_drop, inplace=True)
        df_wd.reset_index(drop=True, inplace=True)

#         print('dropped area: ', areas_to_drop)
        return df_wd
    
    # k formula
    print('----- W_s: ' + str(W_s) + ', W_d: ' + str(W_d) + '-----')
    prob_trips_in_space_k = compute_probability(neighbor_safegraphDF)
    prob_trips_in_space_ws = apply_Ws_formula_optimized(prob_trips_in_space_k, landUse, W_s = W_s)
    prob_trips_in_space_wd = apply_Wd_formula_optimized(prob_trips_in_space_ws, landUse, W_d = W_d)
    
    # Dealing with Empty variables
    for index, row in tqdm(prob_trips_in_space_wd.iterrows(), total = len(prob_trips_in_space_wd), desc = '4) Filling empty values'):
        if row['weekday_Work'] == '{}' or row['weekend_Work'] == '{}':
            area_value = row['area']
            prob_trips_in_space_wd.at[index, 'weekday_Work'] = f"{{'{area_value}': 1.0}}"
            prob_trips_in_space_wd.at[index, 'weekend_Work'] = f"{{'{area_value}': 1.0}}"

    # Remove "Unnamed: 0" if existed.
    if 'Unnamed: 0' in prob_trips_in_space_wd.columns:
        prob_trips_in_space_wd.drop(columns=['Unnamed: 0'], inplace=True)
    
    return prob_trips_in_space_wd
    


# ### A.2.3. fill empty probability

# In[20]:


# 빈것들 채워주는 코드

def fill_values(row):
    """
    Fills empty probability distributions for trip purposes in a row of a DataFrame with alternative values based on a predefined hierarchy of preferences. 
    This ensures that each trip purpose category has a valid probability distribution, either specific or borrowed from a related category.

    Parameters:
    - row (pd.Series): A row from a DataFrame representing trip distributions for various purposes.

    Returns:
    - pd.Series: The input row with empty trip distribution categories filled with alternative values.
    """
    
    # Process 'weekday_Dailycare' column.
    # If there are no probabilities defined for weekday Dailycare, attempt to use the
    # probabilities from 'weekday_Religion' or 'weekday_School' as alternatives.
    if row['weekday_Dailycare'] == '{}':
        if row['weekday_Religion'] != '{}':
            row['weekday_Dailycare'] = row['weekday_Religion']
        elif row['weekday_School'] != '{}':
            row['weekday_Dailycare'] = row['weekday_School']

    if row['weekend_Dailycare'] == '{}':
        if row['weekday_Dailycare'] != '{}':
            row['weekend_Dailycare'] = row['weekday_Dailycare']
        elif row['weekend_Religion'] != '{}':
            row['weekend_Dailycare'] = row['weekend_Religion']
        elif row['weekend_School'] != '{}':
            row['weekend_Dailycare'] = row['weekend_School']
            
    # weekday_Large_shop
    if row['weekday_Large_shop'] == '{}':
        if row['weekday_Etc_shop'] != '{}':
            row['weekday_Large_shop'] = row['weekday_Etc_shop']

    # weekend_Large_shop
    if row['weekend_Large_shop'] == '{}':
        if row['weekday_Large_shop'] != '{}':
            row['weekend_Large_shop'] = row['weekday_Large_shop']
        elif row['weekend_Etc_shop'] != '{}':
            row['weekend_Large_shop'] = row['weekend_Etc_shop']
        elif row['weekday_Etc_shop'] != '{}':
            row['weekend_Large_shop'] = row['weekday_Etc_shop']
            
    # weekend_Etc_shop
    if row['weekend_Etc_shop'] == '{}':
        if row['weekday_Etc_shop'] != '{}':
            row['weekend_Etc_shop'] = row['weekday_Etc_shop']
        elif row['weekend_Large_shop'] != '{}':
            row['weekend_Etc_shop'] = row['weekend_Large_shop']
    
    # weekday_Religion
    if row['weekday_Religion'] == '{}':
        if row['weekday_Dailycare'] != '{}':
            row['weekday_Religion'] = row['weekday_Dailycare']
        elif row['weekday_School'] != '{}':
            row['weekday_Religion'] = row['weekday_School']
            
    # weekend_Religion
    if row['weekend_Religion'] == '{}':
        if row['weekday_Religion'] != '{}':
            row['weekend_Religion'] = row['weekday_Religion']
        elif row['weekend_Dailycare'] != '{}':
            row['weekend_Religion'] = row['weekend_Dailycare']
            
    if row['weekend_School'] == '{}':
        if row['weekend_Dailycare'] != '{}':
            row['weekend_School'] = row['weekend_Dailycare']
        elif row['weekend_Religion'] != '{}':
            row['weekend_School'] = row['weekend_Religion']            
            
    if row['weekend_Meals'] == '{}':
        if row['weekday_Meals'] != '{}':
            row['weekend_Meals'] = row['weekday_Meals']
        elif row['weekend_Etc_shop'] != '{}':
            row['weekend_Meals'] = row['weekend_Etc_shop']              
            
    if row['weekend_Rec_lei'] == '{}':
        if row['weekday_Rec_lei'] != '{}':
            row['weekend_Rec_lei'] = row['weekday_Rec_lei']   
            
    if row['weekend_Serv_trip'] == '{}':
        if row['weekday_Serv_trip'] != '{}':
            row['weekend_Serv_trip'] = row['weekday_Serv_trip']
            
    if row['weekend_Others'] == '{}':
        if row['weekday_Others'] != '{}':
            row['weekend_Others'] = row['weekday_Others']     
            
            
            
    return row


# # Execusion by User

# In[ ]:





# ## data/Origin폴더에 있는 데이터들은 용량이 크니 그냥 Google Drive에 넣어서 공유하기. 
# ## 여기에 있는 데이터들은 Preprocessing NHTS 와 SafeGraph에만 사용하는 것이니 다운받아서 해보려면 해라. 이런 식으로만 주기

# In[ ]:

current_directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_directory)



prob_2020_09_combined = pd.read_excel(dataPath + 'prob_2020_09_combined.xlsx') # Combined Probability of travels from O to D in Sep, 2020
repaired_NHTS = pd.read_csv(dataPath + 'repaired_NHTS.csv') # preprocessed NHTS
trip_mode = pd.read_csv(dataPath + 'trip_mode_prop_all.csv') # trip mode
cbg = gpd.read_file(dataPath + 'cbg_milwaukee.shp') # CBG shp file
network_road = ox.graph_from_place('Milwaukee, Wisconsin, USA', network_type='drive') # road network data

