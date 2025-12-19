#%%
import pyreadstat as ps
from data_prep import DataCleaner
import matplotlib.pyplot as plt
file_path = r"O:\fsw\Data FSW\MedPsy\Brabant Studie\Data merges\DATA CLEANING\Merge September 2023\MERGES\Qualtrics_Labdata_Cytokine_Postpartum_merge_cleaned.sav"
from scipy.stats import ttest_ind, chi2_contingency
import pandas as pd

# Create an instance of DataCleaner with the dataset
cleaner = DataCleaner(file_path)

#%%
data = cleaner.prepare_week28_data()

#%%
# Create an extra column that represents whether a mother had an EDS score equal or over 12
data["PPD"] = data["EPDS_TOT_8wPP"].apply(lambda x: 1 if x >= 10 else 0)

data["Population_12"].replace({'ASIA': 'OTHER', 'CENTRAL AND SOUTH AMERICA': 'OTHER',
'DUTCH/MIXED': 'DUTCH', 'AFRICA': 'OTHER', 'NORTH AMERICA': 'OTHER'}, inplace=True)

data['Education_12'].replace({'Primary school': 'lower',
'Secondary / high school': 'lower', 'Vocational education': 'lower', 'HAVO / VWO': 'lower'},
inplace=True)

data['Maritalstatus_12'].replace({'In a relationship, but not living together': 'other',
'Other, please specify:': 'other', 'Divorced / separated': 'other',
'Single': 'other'}, inplace=True)

 #%%
def calculate_percentage_and_chi2_by_ppd(data, column_name):
    """
    Calculate the percentage of each category in a specified column for PPD and non-PPD groups,
    and perform a chi-square test to compare the distributions.

    Parameters:
    - data: DataFrame containing the dataset.
    - column_name: The name of the column for which to calculate the percentages and perform the chi-square test.

    Returns:
    - A dictionary with percentages for PPD and non-PPD groups, chi-square statistic, and p-value.
    """
    # Calculate the percentage of each category in the specified column for PPD group
    percentages_ppd = data[data["PPD"] == 1][column_name].value_counts(normalize=True) * 100
    
    # Calculate the percentage of each category in the specified column for non-PPD group
    percentages_non_ppd = data[data["PPD"] == 0][column_name].value_counts(normalize=True) * 100
    
    # Create a contingency table for the chi-square test
    contingency_table = pd.crosstab(data[column_name], data["PPD"])
    
    # Perform the chi-square test
    chi2, p_val, dof, expected = chi2_contingency(contingency_table)
    
    return {
        "percentages_PPD": percentages_ppd,
        "percentages_non_PPD": percentages_non_ppd,
        "chi2": chi2,
        "p-value": p_val
    }

# Example usage with the "Work_12" column
results = calculate_percentage_and_chi2_by_ppd(data, "Historyanxiety_12")
print("Percentages for PPD group:")
print(results["percentages_PPD"])

print("\nPercentages for Non-PPD group:")
print(results["percentages_non_PPD"])

print("\nChi-square statistic:")
print(results["chi2"])

print("\nP-value:")
print(results["p-value"])

#%%
from scipy.stats import mannwhitneyu

def calculate_median_and_wilcoxon_by_ppd(data, numeric_column):
    """
    Calculate the median of a numerical variable separately for PPD and non-PPD groups,
    and perform a Wilcoxon rank-sum test (Mann–Whitney U test) to compare the distributions.

    Parameters:
    - data: DataFrame containing the dataset.
    - numeric_column: The name of the numerical column for which to calculate the median and perform the test.

    Returns:
    - A dictionary with the medians for the PPD and non-PPD groups, U-statistic, and p-value.
    """
    # Extract data for each group, dropping NaN values
    ppd_group = data[data["PPD"] == 1][numeric_column].dropna()
    non_ppd_group = data[data["PPD"] == 0][numeric_column].dropna()
    
    # Calculate medians
    median_ppd = ppd_group.median()
    median_non_ppd = non_ppd_group.median()
    
    # Perform Wilcoxon rank-sum test (Mann–Whitney U test)
    u_stat, p_val = mannwhitneyu(ppd_group, non_ppd_group, alternative='two-sided')
    
    return {
        "median_PPD": median_ppd,
        "median_non_PPD": median_non_ppd,
        "U-statistic": u_stat,
        "p-value": p_val
    }

# Example usage with the "BMI_12" column
results = calculate_median_and_wilcoxon_by_ppd(data, "BMI_12")
print("Median for PPD group:")
print(results["median_PPD"])

print("\nMedian for Non-PPD group:")
print(results["median_non_PPD"])

print("\nWilcoxon rank-sum U-statistic:")
print(results["U-statistic"])

print("\nP-value:")
print(results["p-value"])

#%%
from scipy.stats import chi2_contingency
import pandas as pd

def calculate_percentage_and_chi2_onehot_by_ppd(data, categorical_column):
    """
    One-hot encode a categorical variable, calculate the percentage of each one-hot category separately for PPD and non-PPD groups,
    and perform a chi-square test to compare the distributions.

    Parameters:
    - data: DataFrame containing the dataset.
    - categorical_column: The name of the categorical column to one-hot encode, calculate the percentages, and perform the chi-square test.

    Returns:
    - A dictionary where keys are the one-hot encoded categories, and the values are dictionaries containing:
      - percentages for the PPD and non-PPD groups,
      - chi-square statistic, degrees of freedom, and p-value.
    """
    # One-hot encode the categorical column
    onehot_data = pd.get_dummies(data[categorical_column], prefix=categorical_column)

    results = {}

    # For each one-hot encoded category, calculate percentages and perform chi-square test
    for col in onehot_data.columns:
        # Combine the one-hot encoded column with the PPD status
        combined_data = pd.concat([onehot_data[col], data["PPD"]], axis=1)

        # Calculate the percentage of 1's in the specified column for PPD group
        percentages_ppd = combined_data[combined_data["PPD"] == 1][col].mean() * 100

        # Calculate the percentage of 1's in the specified column for non-PPD group
        percentages_non_ppd = combined_data[combined_data["PPD"] == 0][col].mean() * 100

        # Create a contingency table for the chi-square test
        contingency_table = pd.crosstab(combined_data[col], combined_data["PPD"])

        # Perform the chi-square test
        chi2, p_val, dof, expected = chi2_contingency(contingency_table)

        # Store results for each one-hot encoded category
        results[col] = {
            "percentages_PPD": percentages_ppd,
            "percentages_non_PPD": percentages_non_ppd,
            "chi2": chi2,
            "degrees_of_freedom": dof,
            "p-value": p_val
        }

    return results


#%%
def plot_epds_distribution(data, column_name='EPDS_TOT_8wPP'):
    """
    Creates a histogram to show the distribution of the EPDS_TOT_8wPP variable,
    with each bin representing one score and the x-axis labels shown every 5 points.

    Parameters:
    - data: pandas DataFrame, containing the data with the EPDS_TOT_8wPP column.
    - column_name: str, the name of the column to plot (default is 'EPDS_TOT_8wPP').
    """

    if column_name not in data.columns:
        raise ValueError(f"Column '{column_name}' not found in the data.")

    # Extract the data to plot
    epds_values = data[column_name].dropna()

    # Define bins so that each bin represents one score
    min_value = int(epds_values.min())
    max_value = int(epds_values.max())
    bins = range(min_value, max_value + 2)  # +2 to include the last value as a separate bin

    # Set up the plot
    plt.figure(figsize=(7, 6))  # Same width as in the 'pp' step of the previous function
    plt.hist(epds_values, bins=bins, color='#333333', edgecolor='black', align='left')

    # Set labels and title
    plt.xlabel('EDS Total Score at 8-10 Weeks Postpartum', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)

    # Set consistent font size for tick labels and set x-ticks every 5 points
    xticks = range(min_value, max_value + 1, 5)  # Labels every 5 points
    plt.xticks(xticks, fontsize=12)
    plt.yticks(fontsize=12)

    # Show the plot
    plt.show()


#%%
plot_epds_distribution(data)

#%%
# Print percentage of mothers with PPD
print(f"Percentage of mothers with PPD: {data['PPD'].mean() * 100:.2f}%")

#%%
data['PPD'].value_counts()

#%%
columns_bi = ['Lifeevent_12', 'Lifeevent_20', 'Lifeevent_28',
'Historydepression_12', 'Eczema_12', 'Smoking_12', 'Rheumatism_12',
'Diabetes_12','Plannedpreg_12']

columns_onehot = ['Treatment_12',
'Work_12', 'GDMmother_12', 'Workpartner_12', 'Population_12',
'Sexdetermination_12', 'Education_12', 'Maritalstatus_12', 'Alcohol_12']	

columns_cont = ['SCL90_TOT_12', 'SCL90_TOT_20', 'SCL90_TOT_28',
'EDS_TOT_12', 'EDS_TOT_20', 'EDS_TOT_28',
'TPDS_TOT_12', 'TPDS_TOT_20', 'TPDS_TOT_28', 'IL6H_12', 'IL6H_20',
'IL6H_28',
'TFMQ_SF_Nonreacting_TOT_20', 'TFMQ_SF_Nonjudging_TOT_20',
'TFMQ_SF_Awareness_TOT_20', 'DS14_NA_TOT_12',
'DS14_SI_TOT_12', 'MSPSS_Family_TOT_20',
'MSPSS_Friends_TOT_20',
'BFI_2S_Extraversion_TOT_28',
'BFI_2S_Agreeableness_TOT_28', 'BFI_2S_Conscientiousness_TOT_28',
'BFI_2S_NE_TOT_28', 'BFI_2S_Openmindedness_TOT_28',
'BMI_12', 'BMI_20', 'BMI_28', 'Complaints_13_12',
'Complaints_13_20', 'Complaints_13_28', 'TPDS_TOT_12',
'TPDS_TOT_20', 'TPDS_TOT_28', 'Scale_forgetfulness_12',
'Scale_forgetfulness_20', 'Scale_forgetfulness_28', 'TRFE_12',
'Complaints_6_12', 'Complaints_6_20', 'Complaints_6_28',
'VBBA_Supervisor_TOT_12', 'VBBA_Supervisor_TOT_28',
'VBBA_Colleagues_TOT_12', 'VBBA_Colleagues_TOT_28', 'NBNPH_12',
'NBNPH_20', 'NBNPH_28', 'Complaints_4_12', 'Complaints_4_20',
'Complaints_4_28', 'SMUmotives_17_12', 'Complaints_17_12',
'Complaints_17_20', 'Complaints_17_28', 'ATPOH_12',
'ATPOH_20', 'ATPOH_28', 'Work_Engagement_TOT_12',
'Work_Engagement_TOT_28', 'Work_Burnout_TOT_12',
'Work_Burnout_TOT_28', 'TSHH_12', 'TSHH_20', 'TSHH_28',
'Scale_concentrating_12', 'Scale_concentrating_20',
'Scale_concentrating_28', 'SMUmotives_18_12', 
'BSMAS_TOT_12', 'BSMAS_TOT_20', 'BSMAS_TOT_28',
'Age_12']

#%%
# Combine binary and one-hot columns for chi-square testing
columns_cat = columns_bi + columns_onehot

import pyreadstat as ps
import pandas as pd
from scipy.stats import chi2_contingency
import numpy as np

#%%
wilcoxon_results = {}
for var in columns_cont:
    results = calculate_median_and_wilcoxon_by_ppd(data, var)
    wilcoxon_results[var] = results

# Convert the results to a DataFrame for easier viewing
wilcoxon_df = pd.DataFrame.from_dict(wilcoxon_results, orient='index')

print("\nWilcoxon Rank-Sum Test Results for Continuous Variables:")
print(wilcoxon_df)

#%%
# Import necessary libraries
from scipy.stats import chi2_contingency
import pandas as pd

# Ensure that the function calculate_percentage_and_chi2_by_ppd is also defined
def calculate_percentage_and_chi2_by_ppd(data, column_name):
    """
    Calculate the percentage of each category in a specified column for PPD and non-PPD groups,
    and perform a chi-square test to compare the distributions.

    Parameters:
    - data: DataFrame containing the dataset.
    - column_name: The name of the column for which to calculate the percentages and perform the chi-square test.

    Returns:
    - A dictionary with percentages for PPD and non-PPD groups, chi-square statistic, degrees of freedom, and p-value.
    """
    # Calculate the percentage of each category in the specified column for PPD group
    percentages_ppd = data[data["PPD"] == 1][column_name].value_counts(normalize=True) * 100

    # Calculate the percentage of each category in the specified column for non-PPD group
    percentages_non_ppd = data[data["PPD"] == 0][column_name].value_counts(normalize=True) * 100

    # Create a contingency table for the chi-square test
    contingency_table = pd.crosstab(data[column_name], data["PPD"])

    # Perform the chi-square test
    chi2, p_val, dof, expected = chi2_contingency(contingency_table)

    return {
        "percentages_PPD": percentages_ppd,
        "percentages_non_PPD": percentages_non_ppd,
        "chi2": chi2,
        "degrees_of_freedom": dof,
        "p-value": p_val
    }

# Initialize a dictionary to store the results
chi2_results = {}

# Analyze binary variables using calculate_percentage_and_chi2_by_ppd
for col in columns_bi:
    result = calculate_percentage_and_chi2_by_ppd(data, col)
    chi2_results[col] = {
        'chi2': result['chi2'],
        'degrees_of_freedom': result['degrees_of_freedom'],
        'p-value': result['p-value'],
        'percentages_PPD': result['percentages_PPD'].to_dict(),
        'percentages_non_PPD': result['percentages_non_PPD'].to_dict()
    }

# Analyze one-hot encoded variables using calculate_percentage_and_chi2_onehot_by_ppd
for col in columns_onehot:
    results = calculate_percentage_and_chi2_onehot_by_ppd(data, col)
    for category, res in results.items():
        chi2_results[category] = {
            'chi2': res['chi2'],
            'degrees_of_freedom': res['degrees_of_freedom'],
            'p-value': res['p-value'],
            'percentages_PPD': res['percentages_PPD'],
            'percentages_non_PPD': res['percentages_non_PPD']
        }

# Convert the results dictionary to a DataFrame for easier viewing
chi2_results_df = pd.DataFrame.from_dict(chi2_results, orient='index')

# Display the results
print("\nChi-Square Test Results for Categorical Variables:")
print(chi2_results_df)

#%%
# calculate the correlation matrix of EDS_TOT_12, EDS_TOT_20, EDS_TOT_ 28, EPDS_TOT_8wPP
correlation_matrix = data[['EDS_TOT_12', 'EDS_TOT_20', 'EDS_TOT_28',
'EPDS_TOT_8wPP', 'GAD_7_TOT_8wPP', 'SCL90_TOT_8wPP']].corr(method='spearman')

#%%
correlation_matrix

#%%
import pyreadstat as ps
#path = r"O:\fsw\Data FSW\MedPsy\Brabant Studie\DATA\DATA CLEANING\Merge September 2023\MERGES\Qualtrics_Labdata_Cytokine_Postpartum_merge_cleaned.sav"
path = r"O:\fsw\Data FSW\MedPsy\Brabant Studie\DATA\DATA CLEANING\Merge September 2023\MERGES\Qualtrics_Labdata_Cytokine_Postpartum_Obstetrics_merge_cleaned.sav"
test_data, meta = ps.read_sav(path)
#%%
# create a correlation matrix of 8wPP individual EPDS items and GAD_7 items
epds_items = [f'EPDS_{i}_8wPP' for i in range(1, 11)]
gad_items = [f'GAD_7_{i}_8wPP' for i in range(1, 8)]

#%%
correlation_matrix = test_data[epds_items + gad_items].corr(method='spearman')

#%%
correlation_matrix

#%%
# correlation matric of EPDS items and GAD_7_TOT_8wPP
correlation_matrix = test_data[epds_items + ['GAD_7_TOT_8wPP', 'EPDS_TOT_8wPP']].corr(method='spearman')

#%%
correlation_matrix

#%%
participants_to_check = [747, 811, 882, 894, 1076, 1155]

#%%
# Example for a specific model
model_results = results['model_name']  # Replace 'model_name' with the actual model key

participant_ids = model_results['participant_ids']
predictions = model_results['predictions']
y_test = model_results['y_test']

# You can create a DataFrame to view the results more easily
results_df = pd.DataFrame({
    'Participant_ID': participant_ids,
    'True_Value': y_test,
    'Prediction': predictions
})

print(results_df)