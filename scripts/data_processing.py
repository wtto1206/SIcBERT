import pandas as pd
import re
import random

"""
Script for cleaning and processing dataset.

This script loads a dataset, processes the text columns, removes unwanted metadata, creates new neutral sentence pairs, 
and formats the data into a structured format ready for analysis.

Usage:
    python clean_dataset.py

Input:
    - data.csv: Original dataset containing multiple columns including text fields.

Output:
    - cleaned_data.csv: Processed and cleaned dataset.
"""


# Function to generate 500 correct neutral sentence pairs
def create_neutral_pairs(df, num_pairs=500, random_state=42):
    random.seed(random_state)
    new_rows = []
    used_pairs = set()

    # Remove existing neutral rows
    df = df[df['label'] != 'neutral'].reset_index(drop=True)

    # Take unique combinations from premise + scale
    premise_pool = df[['Scale', 'premise']].drop_duplicates().reset_index(drop=True)

    while len(new_rows) < num_pairs:
        i, j = random.sample(range(len(premise_pool)), 2)
        if i == j:
            continue

        premise_i = premise_pool.loc[i, 'premise']
        premise_j = premise_pool.loc[j, 'premise']

        if (premise_i, premise_j) in used_pairs or (premise_j, premise_i) in used_pairs:
            continue

        new_row = {
            'Scale': premise_pool.loc[i, 'Scale'],
            'premise': premise_i,
            'alternative': premise_pool.loc[j, 'Scale'],
            'statement': premise_j,
            'value_mean': 3.0,  # Optional, just for completeness
            'label': 'neutral'
        }

        new_rows.append(new_row)
        used_pairs.add((premise_i, premise_j))

    df_neutral = pd.DataFrame(new_rows)
    df_extended = pd.concat([df, df_neutral], ignore_index=True)

    return df_extended

def clean_dataset():
    # Load the original dataset
    data = pd.read_csv('/home/wtto/Documents/HHU/fourth_semester/NLI_AP/data.csv')  # Change the path to the dataset

    # Create a new column 'first_ten_words' which contains the first ten words of the 'text' column
    data['first_ten_words'] = data['text'].str.split().str[:10].apply(lambda x: ' '.join(x))
    
    # Group by the 'first_ten_words' and calculate the mean of the 'value' column
    mean_values = data.groupby('first_ten_words')['value'].mean().reset_index()

    # Merge the mean values back into the filtered dataframe
    data = pd.merge(data, mean_values, on='first_ten_words', suffixes=('', '_mean'))

    # Drop unnecessary columns
    data = data.drop(columns=['QID', 'Group', 'Subject', 'text1', 'value', 'attributive', 'predicative'])

    # Drop duplicates, keeping only one row per group of the same first ten words
    data = data.drop_duplicates(subset=['first_ten_words'])

    # Optionally, drop the 'first_ten_words' column if you don't need it anymore
    data = data.drop(columns=["first_ten_words"])

    # Lowercase the 'text2' column
    data['text2'] = data['text2'].str.lower()

    # Remove everything between "<" and ">" in the 'text2' column, including the signs themselves
    data['text2'] = data['text2'].str.replace(r'<.*?>', '', regex=True)

    # Rename columns to meaningful names
    data.rename(columns={"text": "premise", "text2": "statement"}, inplace=True)

    # Add the label column based on the value_mean column
    data['label'] = data['value_mean'].apply(lambda x: 'entailment' if x >= 4 else 'contradiction')

    # Drop the 'value_mean' column as it's no longer needed
    data.drop(columns=['value_mean'], inplace=True)

    # Generate the new dataset with 500 additional neutral pairs
    df_extended = create_neutral_pairs(data, num_pairs=500, random_state=42)

    # Drop the 'value_mean' column as it's no longer needed
    df_extended.drop(columns=['value_mean'], inplace=True)

    # Save the cleaned dataset
    df_extended.to_csv('evaluation_dataset.csv', index=False)


    print("Data cleaning and labeling complete. Saved as 'evaluation_dataset.csv'.")

if __name__ == "__main__":
    clean_dataset()
