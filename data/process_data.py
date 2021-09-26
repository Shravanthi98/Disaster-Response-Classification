# Import all the necessary libraries.
import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Function to load the given data sets.
    Arguments:
    messages_filepath: Path to the location of "disaster_messages.csv" file
    categories_filepath: Path to the location of "disaster_categories.csv" file

    Returns a Data frame with merged data sets.
    """
    # Read in the messages and categories files.
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    # Merge the data sets on common ID.
    df = messages.merge(categories, on='id', how='inner')
    return df

def clean_data(df):
    """
    Function used to clean the given data.
    Arguments:
    df: Data frame with messages and respective categories.

    Returns a clean data frame.
    """
    # Split the categories and expand into columns.
    categories = df['categories'].str.split(';', expand=True)
    # Select the first row of the categories dataframe
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x.rstrip('-0 1'))
    # Rename the columns of `categories`
    categories.columns = category_colnames
    for column in categories:
        # Set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x.split('-')[1])
    
        # Convert the column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
    # Drop the unnecessary columns
    df.drop(['categories', 'original'], axis=1, inplace=True)
    # Concatenate the newly formed categories to the existing data frame
    df = pd.concat([df, categories], axis=1)
    # Drop all the duplicates
    df.drop_duplicates(subset=['message'], inplace=True)
    # Dropping this column as it doesn't change values
    df.drop('child_alone', axis=1, inplace=True)
    # Deleting the row values where the 'related' column is greater than 1 - could be error values.
    final_df = df[df['related']!=2]
    return final_df

def save_data(df, database_filename):
    """
    Function to save the data frame as a SQL Data base.
    Arguments:
    df: Data frame to be saved
    database_filename: Provide a name for the SQL Database.
    """
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('Disaster', engine, index=False, if_exists='replace')  


def main():
    # Provide command line arguments for the paths to the datasets and a path to save the data base file.
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
