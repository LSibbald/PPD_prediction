#%%
import os
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from data_prep import DataCleaner

def fill_missing_categorical(data, columns_cat):
    """Fill missing categorical values with 'missing'."""
    # Check if all columns in columns_cat exist in the DataFrame
    missing_cols = [col for col in columns_cat if col not in data.columns]
    if missing_cols:
        raise ValueError(f"The following columns are not in the DataFrame: {missing_cols}")

    # Debug: Print initial state
    print(f"Initial data shape: {data.shape}")
    print(f"Initial columns: {data.columns.tolist()}")

    # Initialize the imputer
    imputer = SimpleImputer(strategy='constant', fill_value='missing')

    # Impute missing values for each column in columns_cat
    for col in columns_cat:
        if col in data.columns:
            data[col] = imputer.fit_transform(data[[col]]).ravel()

    # Debug: Print final state
    print(f"Data shape after imputation: {data.shape}")
    print(f"Columns after imputation: {data.columns.tolist()}")

    return data

# define a function to OneHotEncode the categorical columns 
def one_hot_encode(data, columns_cat):
    """OneHotEncode the categorical columns."""
    # Initialize the OneHotEncoder
    one_enc = OneHotEncoder()

    # Fit and transform the OneHotEncoder on the categorical columns
    encoded_features = one_enc.fit_transform(data[columns_cat]).toarray()

    # Get the column names for the one-hot encoded features
    column_names = one_enc.get_feature_names_out(columns_cat)

    # Create a DataFrame with the one-hot encoded features  
    encoded_df = pd.DataFrame(encoded_features, columns=column_names, index=data.index)

    # Drop the original categorical columns from the original DataFrame
    data.drop(columns=columns_cat, inplace=True)

    # Concatenate the original DataFrame with the one-hot encoded DataFrame
    encoded_data = pd.concat([data, encoded_df], axis=1)

    return encoded_data

def replace_zeros_with_nan(data, columns_cat):
    """Replace 0's in columns starting with a column in columns_cat with np.nan if columns ending on 'missing' contain a 1."""
    for cat in columns_cat:
        missing_col = f"{cat}_missing"
        if missing_col in data.columns:
            for col in data.columns:
                if col.startswith(cat) and col != missing_col:
                    data.loc[data[missing_col] == 1, col] = np.nan
    
    # Drop the columns ending on 'missing'
    data = data.drop(columns=[col for col in data.columns if col.endswith('missing')])

    return data

#%%
def one_hot_encode_with_imputation(data, columns_cat, k_values, file_prefix, directory_path):
    """Perform one-hot encoding, handle missing values, and apply KNN imputation."""
    # Fill missing categorical values
    data = fill_missing_categorical(data, columns_cat)
    
    # One-hot encode the categorical columns
    data = one_hot_encode(data, columns_cat)
    
    # Replace 0's with np.nan where the 'missing' indicator is 1
    data = replace_zeros_with_nan(data, columns_cat)
    
    # Split the dataframe into training and test sets (80% train, 20% test)
    train_df, test_df = train_test_split(data, test_size=0.2, random_state=42)
    
    for k in k_values:
        # Initialize the KNN Imputer with the current k value
        imputer = KNNImputer(n_neighbors=k)
        
        # Fit the imputer on the training set and transform the training set
        imputed_train_data = imputer.fit_transform(train_df)
        imputed_train_df = pd.DataFrame(imputed_train_data, columns=train_df.columns, index=train_df.index)
        
        # Transform the test set using the fitted imputer (without fitting it again)
        imputed_test_data = imputer.transform(test_df)
        imputed_test_df = pd.DataFrame(imputed_test_data, columns=test_df.columns, index=test_df.index)
        
        # Create directory if it does not exist
        os.makedirs(directory_path, exist_ok=True)
        
        # Save the imputed training dataset
        train_filename = f"{directory_path}/{file_prefix}_train_knn_imputed_k{k}.csv"
        imputed_train_df.to_csv(train_filename, index=True)
        print(f"Imputed training dataset saved to {train_filename}")
        
        # Save the imputed test dataset
        test_filename = f"{directory_path}/{file_prefix}_test_knn_imputed_k{k}.csv"
        imputed_test_df.to_csv(test_filename, index=True)
        print(f"Imputed test dataset saved to {test_filename}")

    return train_df, test_df

#%%
# Path to dataset
file_path = r"O:\fsw\Data FSW\MedPsy\Brabant Studie\DATA\DATA CLEANING\Merge September 2023\MERGES\Qualtrics_Labdata_Cytokine_Postpartum_Obstetrics_merge_cleaned.sav"

# Directory where you want to save the imputed datasets
save_directory = r"O:\fsw\Data FSW\MedPsy\Brabant Studie\DATA\DATA CLEANING\Project_Lisette"

# Create an instance of DataCleaner with the dataset
cleaner = DataCleaner(file_path)

# Define a list of k values for which you want to perform KNN imputation
k_values = [3, 5, 7]

#%%
# WEEK 12
week12_data = cleaner.prepare_week12_data()
columns_cat_12 = ["Maritalstatus_12", "Education_12", "Population_12", "Sexdetermination_12", "Sexpreference_12",
                      "Locationdelivery_12", "Painmanagement_12", "Work_12", "Worksituation_12", "Workpostpreg_12",
                      "Workpartner_12", "Hoursworkpartner_12", "Breadwinner_12", "Alcohol_12", "Smokingpartner_12",
                      "BPmother_12", "Diabetesmother_12", "CVDmother_12", "GDMmother_12", "BPpregmother_12",
                      "Treatment_12", "Nausea_weeks_12"]

#%%
one_hot_encode_with_imputation(week12_data, columns_cat_12, k_values, "week12", save_directory)

#%%
# inspect the imputed data
week12_test_k3 = pd.read_csv(os.path.join(save_directory, "week12_test_knn_imputed_k3.csv"), index_col=0)

#%%
# WEEK 20
week20_data = cleaner.prepare_week20_data()
columns_cat_week20 = columns_cat_12 + ['Pregguidance_20', 'Gynaecologist_20', 'Particularityecho_20', 'Healthinsurance_20',
                      'Alcohol_20', 'Smokingpartner_20', 'Nausea_weeks_20']

one_hot_encode_with_imputation(week20_data, columns_cat_week20, k_values, "week20", save_directory)

#%%
week20_test_k3 = pd.read_csv(os.path.join(save_directory, "week20_test_knn_imputed_k3.csv"), index_col=0)

#%%
# WEEK 28
week28_data = cleaner.prepare_week28_data()
columns_cat_week28 = columns_cat_week20 + ['Pregguidance_28', 'Gynaecologist_28', 'Nausea_weeks_28', 'Work_28', 'Alcohol_28',
                      'Smokingpartner_28']

one_hot_encode_with_imputation(week28_data, columns_cat_week28, k_values, "week28", save_directory)

#%%
# POSTPARTUM
pp_data = cleaner.prepare_obs_data()
columns_cat_postpartum = columns_cat_week28 + ['Season_birth_8wPP', 'Pregguidance_8wPP', 'Gynaecologist_8wPP', 'Problemspreg_8wPP', 'Delivery_8wPP',
                          'Spinalpuncture_8wPP', 'Checkup_8wPP', 'Checkupresults_8wPP', 'Nutrition_8wPP',
                          'Workpostpreg_8wPP', 'Workchanges_8wPP', 'Workpartner_8wPP', 'Hoursjobpartner_8wPP',
                          'Breadwinner_8wPP', 'Opinioncrying_8wPP', 'Alcohol_8wPP', 'Smokingpartner_8wPP',
                          'Partus_locatie_OBS', 'Breken_van_de_vliezen_OBS', 'Baringsuitkomst_OBS',
                          'Ligging_kind_OBS', 'Meconium_OBS', 'Practice_OBS']

one_hot_encode_with_imputation(pp_data, columns_cat_postpartum, k_values, "pp", save_directory)
