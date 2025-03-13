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


# Function to generate 500 new neutral sentence pairs
def create_neutral_pairs(df, num_pairs=500, random_state=42):
    new_rows = []

    # Randomly sample 500 rows (or all rows if there are fewer)
    sampled_rows = df.sample(n=min(num_pairs, len(df)), random_state=random_state)

    # Shuffle the "alternative" column while ensuring different adjectives
    shuffled_alternatives = df["alternative"].sample(frac=1, random_state=random_state).values

    for i, (idx, row) in enumerate(sampled_rows.iterrows()):
        orig_adj = row["alternative"]
        new_adj = random.choice(shuffled_alternatives)

        while new_adj == orig_adj:  # Ensure the new adjective is different
            new_adj = random.choice(shuffled_alternatives)

        # Modify the statement column
        new_statement = row["statement"].replace(f"but not {orig_adj}", f"but not {new_adj}")

        # Create a new row
        new_row = {
            "Scale": row["Scale"],
            "premise": row["premise"],
            "alternative": new_adj,
            "statement": new_statement,
            "value_mean": 3.0,
            "label": "neutral"
        }
        new_rows.append(new_row)

        # Stop if we reach 500 new pairs
        if len(new_rows) >= num_pairs:
            break

    # Convert new rows to DataFrame and append to the original dataset
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