import pandas as pd
import numpy as np

# import splitting functions
from sklearn.model_selection import train_test_split

def prep_iris(df):
    """
    write meaningful docstrings
    """
    df = df.drop(columns=['species_id','measurement_id'])
    df = df.rename(columns={"species_name":'species'})
    
    return df

def clean_titanic(df):
    """
    students - write docstring
    """
    #drop unncessary columns
    df = df.drop(columns=['embarked', 'age','deck', 'class'])
    
    #made this a string so its categorical
    df.pclass = df.pclass.astype(object)
    
    #filled nas with the mode
    df.embark_town = df.embark_town.fillna('Southampton')
    
    return df

def prep_telco(df):
    '''
    Prepare the Telco data by removing unnecessary columns and handling missing values.

    Args:
    - df (DataFrame): Input DataFrame containing Telco data.

    Returns:
    - DataFrame: Processed Telco data with specified columns dropped and missing values handled.

    This function removes the columns 'payment_type_id', 'internet_service_type_id', and 'contract_type_id'
    from the input DataFrame. Additionally, it replaces any empty strings in the 'total_charges' column with '0.0'.

    Example:
        telco_data = get_telco_data()
        prepped_telco = prep_telco(telco_data)
    '''
    df = df.drop(columns = ['payment_type_id','internet_service_type_id','contract_type_id'])
    df.total_charges = df.total_charges.str.replace(' ', '0.0')
    
    return df


def telco_encoded(train, validate, test):
    """
    One-hot encodes categorical columns in the given DataFrames (train, validate, and test).

    Parameters:
    - train (pd.DataFrame): The training dataset.
    - validate (pd.DataFrame): The validation dataset.
    - test (pd.DataFrame): The test dataset.

    Returns:
    List of Encoded DataFrames:
    - train_encoded (pd.DataFrame): Encoded training dataset.
    - validate_encoded (pd.DataFrame): Encoded validation dataset.
    - test_encoded (pd.DataFrame): Encoded test dataset.

    This function performs one-hot encoding on categorical columns in the provided DataFrames,
    excluding 'customer_id' and 'total_charges'. Dummy variables are created for categorical
    columns using pd.get_dummies, and the original columns are dropped from the DataFrames.

    Example:
    train_encoded, validate_encoded, test_encoded = telco_encoded(train_df, validate_df, test_df)
    """
    encoded_dfs = []
    for df in [train, validate, test]:
        df_encoded = df.copy()
        for col in df.columns:
            if col == 'customer_id':
                continue
            if col == 'total_charges':
                continue
            elif df[col].dtype == 'O':  
                df_dummies = pd.get_dummies(df[col], prefix=col, drop_first=True).astype(int)
                df_encoded = df_encoded.join(df_dummies).drop(columns=[col])
        encoded_dfs.append(df_encoded)
    return encoded_dfs


def splitting_data(df, col):
    '''
    Prepare the Telco dataset by cleaning and transforming the data.

    Parameters:
    - df (DataFrame): The input DataFrame containing Telco data.

    Returns:
    - DataFrame: The cleaned and transformed Telco DataFrame.

    Steps:
    1. Drop unnecessary columns: 'payment_type_id', 'internet_service_type_id', 'contract_type_id'.
    2. Replace any empty spaces in 'total_charges' with '0.0'.
    
    Example:
        telco_data = pd.read_csv('telco.csv')
        cleaned_telco = prep_telco(telco_data)
    '''

    #first split
    train, validate_test = train_test_split(df,
                     train_size=0.6,
                     random_state=123,
                     stratify=df[col]
                    )
    
    #second split
    validate, test = train_test_split(validate_test,
                                     train_size=0.5,
                                      random_state=123,
                                      stratify=validate_test[col]
                        
                                     )
    return train, validate, test

