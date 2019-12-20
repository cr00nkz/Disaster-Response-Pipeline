# import libraries
import pandas as pd
from sqlalchemy import create_engine

def load_data():
    """Load the messages and categories csv files from the folder it is in

    RETURNS: messages dataframe, categories dataframe"""

    # load messages datasets
    messages = pd.read_csv("messages.csv")
    categories = pd.read_csv("categories.csv")

    return messages, categories


def prep_and_clean(messages, categories):
    """Take the messages and categories dataframes and performs the following actions
    * Merge them into one
    * Convert the semicolon separated categories into columns
    * Fill the new columns accordingly
    * Drop duplicates from the data

    INPUT: messages - The messages dataframe
           categories - the categories dataframe

    RETURNS: The cleaned up dataframe"""

    df = pd.merge(messages, categories, on="id") # Merge the data

    # Split into seperate columns copying the data into the rows
    categories = df['categories'].str.split(";", expand = True) 

    #region column name change
    # select the first row of the categories dataframe
    row = categories.iloc[0] 
    # use this row to extract a list of new column names for categories.
    category_colnames = row.str.split(";") 

    # Iterate through each category and clean it while doing so
    # Note, that we are removing the last two characters (-0 or -1)
    new_cat_list = []
    for cat in category_colnames:
        new_cat_list.append(cat[0][:-2])

    # Use the new names to rename our columns
    categories.columns = new_cat_list
    #endregion 


    #Convert row content to 0 or 1
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str.replace(column + "-", "")
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])

    # drop the original categories column from `df`
    df.drop(columns=["categories"], inplace=True)

    # concatenate the original dataframe with the new `categories` dataframe
    df[categories.columns] = categories
    
    # remove duplicates
    print("Removing {} duplicates in the dataframe."
        .format(len(df) - len(df.drop_duplicates())))
    df.drop_duplicates(inplace=True)

    return df


def save_to_db(df, tablename="DISASTER_DATA", should_replace=True):
    """Save the dataframe to a SQLite table. 
    Allows to specify the tablename and the action to perform, 
    in case the table exists

    INPUT: df - The dataframe to save to the database
           tablename - The table to save the dataframe into 
                (defaulted to DISASTER_DATA)
           should_replace - In case the table exists, should it be replaced? 
                (defaulted to True)"""
    
    engine = create_engine('sqlite:///ETL_Disaster.db')
    df.to_sql('DISASTER_DATA', engine, index=False, if_exists="replace")
    print("database: {} - tablename {}".format("ETL_Disaster.db", tablename))


if __name__ == "__main__":
    """Load the data, clean the data, save the data"""
    messages, categories = load_data() # Load data
    print("Loaded data")
    df = prep_and_clean(messages, categories) # Clean data
    print("Cleaned data")
    save_to_db(df) # Save data
    print("Saved data")