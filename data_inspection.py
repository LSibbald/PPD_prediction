#%%
import pyreadstat as ps
#path = r"O:\fsw\Data FSW\MedPsy\Brabant Studie\DATA\DATA CLEANING\Merge September 2023\MERGES\Qualtrics_Labdata_Cytokine_Postpartum_merge_cleaned.sav"
path = r"O:\fsw\Data FSW\MedPsy\Brabant Studie\DATA\DATA CLEANING\Merge September 2023\MERGES\Qualtrics_Labdata_Cytokine_Postpartum_Obstetrics_merge_cleaned.sav"
test_data, meta = ps.read_sav(path)

#%%
# print all columns names ending in _12
def print_columns(ending='_12'):
    for column in test_data.columns:
        if column.endswith(ending):
            print(column)
            

#%%
print_columns(ending='_20')        

#%%
# count non NaN values
def count_nan_values(variable_name):
    return test_data[variable_name].isna().sum()

#%%
count_nan_values('BMI_OBS')

#%%
import matplotlib.pyplot as plt

# create function that plots a histogram of a variable
def plot_histogram(variable_name):
    test_data[variable_name].hist()
    plt.title(variable_name)
    plt.show()

#%%
plot_histogram('BMI_Start_gewicht_OBS')

#%%
# print value counts
test_data['BMI_Start_gewicht_OBS'].value_counts()

#%%
# print value counts for Problemsprevpreg_NA_12
test_data['IL1B_FI_12'].value_counts()

#%%
# print n rows in test_data
test_data.shape[0]

#%%
print_columns('_20')

#%%
week20_columns = [col for col in test_data.columns if '_20' in col]

# Add 'EPDS_TOT_8wPP'and 'RecipientFirstName' to the list
week20_columns.extend(['EPDS_TOT_8wPP','RecipientFirstName'])
        
# Subset the DataFrame
week20_data = test_data[week20_columns]

# Rename 'RecipientFirstName' to 'Participant'
week20_data = week20_data.rename(columns={'RecipientFirstName': 'Participant'})

# Reset index
week20_data = week20_data.set_index('Participant')

#%%
# Check the data types of these columns
perfectionism_vars = [
            'Perfectionism_push_20', 'Perfectionism_standards_20', 'Perfectionism_failure_20',
            'Perfectionism_judge_20', 'Perfectionism_striving_20', 'Perfectionism_careful_20',
            'Perfectionism_ambition_20'
        ]

test_data[perfectionism_vars].dtypes
#%%
def print_value_labels(meta, column_name):
    # Access the value labels dictionary from the metadata object
    value_labels = meta.variable_value_labels
    
    # Check if the column exists in the metadata and if it has value labels
    if column_name in meta.column_names and column_name in value_labels:
        print(f"Value labels for column '{column_name}':")
        for value, label in value_labels[column_name].items():
            print(f"{value}: {label}")
    else:
        print(f"No value labels found for column '{column_name}'.")

#%%
print_columns(ending='_8wPP') 

#%%
print_columns(ending='_OBS')

#%%
count_nan_values('Healthinsurance_20')

#%%
plot_histogram('PRS_TOT_8wPP')

#%%
print_value_labels(meta, 'Healthinsurance_20')

#%%
test_data['Healthinsurance_20'].value_counts()

#%%
columns_to_keep = ['Pregguidance_8wPP', 'Gynaecologist_8wPP', 'Problemspreg_8wPP', 'Pregduration_8wPP',
                   'Genderbaby_8wPP', 'Weightbaby_8wPP', 'Delivery_8wPP', 'Episiotomy_8wPP',
                   'Rupture_8wPP', 'Durationdilated_8wPP', 'Durationpush_8wPP', 'Spinalpuncture_8wPP',
                   'Painkiller_8wPP', 'Planneddeliv_8wPP', 'Weightgain_8wPP', 'Complaintspreg_8wPP',
                   'Complaintsdeliv_8wPP', 'Checkup_8wPP', 'Checkupresults_8wPP', 'Contraception_8wPP',
                   'Menstruation_8wPP', 'Doctorvisitafterdeliv_8wPP', 'Illness_8wPP',
                   'Medicationafterdeliv_8wPP', 'Medication_unreg_afterdeliv_8wPP', 'Nutrition_8wPP',
                   'Breastfeeding_8wPP', 'Doctorvisitbaby_8wPP', 'Medicationbaby_8wPP', 'Frequencycrying_8wPP',
                   'Dailycrying_8wPP', 'Opinioncrying_8wPP', 'Cryhour_8wPP', 'Comforting_8wPP',
                   'Hourscrying_8wPP', 'Defecationbaby_8wPP', 'Refluxbaby_8wPP', 'Rashbaby_8wPP',
                   'Coldbaby_8wPP', 'Coughingbaby_8wPP', 'Weightbaby_healthcentre_8wPP',
                   'Lengthbaby_healthcentre_8wPP', 'Peculiarities_8wPP', 'Alcohol_8wPP', 'Smoking_8wPP',
                   'Smokingpartner_8wPP', 'Lifeevent_8wPP', 'Supplements_8wPP', 'Medicationpreg_8wPP',
                   'Medication_unreg_preg_8wPP', 'Doctorvisitlastweeks_8wPP', 'Hospitalisation_8wPP',
                   'Partner_8wPP', 'Work_8wPP', 'Maternityleave_8wPP', 'Plannedmatleave_8wPP', 'Workpostpreg_8wPP',
                   'Returntowork_8wPP', 'Workduringpreg_8wPP', 'Workchanges_8wPP', 'Workpartner_8wPP',
                   'Hoursjobpartner_8wPP', 'Breadwinner_8wPP', 'Absence_work_8wPP', 'TPDS_PI_TOT_8wPP',
                   'BMI_8wPP', 'DAS_Satisfaction_TOT_8wPP', 'SCL90_TOT_8wPP', 'MSPSS_Family_TOT_8wPP',
                   'MSPSS_Friends_TOT_8wPP', 'ECR_SF_Anxiety_TOT_8wPP', 'ECR_SF_Avoidance_TOT_8wPP',
                   'Season_birth_8wPP', 'Weekend_birth_8wPP', 'PPBS_TOT_8wPP', 'PRS_TOT_8wPP',
                   'DERS16_Clarity_TOT_8wPP', 'DERS16_Goals_TOT_8wPP', 'DERS16_Impulse_TOT_8wPP',
                   'DERS16_Strategies_TOT_8wPP', 'DERS16_Nonacceptance_TOT_8wPP', 'GAD_7_TOT_8wPP'
                   'OBVL_Parent_Child_Relationship_8wPP', 'OBVL_Parenting_8wPP', 'OBVL_Role_Limitation_8wPP']

columns_cat = ['Pregguidance_8wPP', 'Gynaecologist_8wPP', 'Problemspreg_8wPP', 'Delivery_8wPP',
               'Spinalpuncture_8wPP', 'Checkup_8wPP', 'Checkupresults_8wPP', 'Nutrition_8wPP',
               'Workpostpreg_8wPP', 'Workchanges_8wPP', 'Workpartner_8wPP', 'Hoursjobpartner_8wPP',
               'Breadwinner_8wPP',  'Opinioncrying_8wPP', 'Alcohol_8wPP',
               'Smokingpartner_8wPP']

nog_checken = ['Weekschildbirth_8wPP']

labels_mappen = ['Problemspreg_8wPP', 'Opinioncrying_8wPP', 'Defecationbaby_8wPP',
                 'Smokingpartner_8wPP', 'Doctorvisitlastweeks_8wPP', 'Season_birth_8wPP']

#%%
count_nan_values('BMI_Start_gewicht_OBS')

#%%
plot_histogram('BMI_Start_gewicht_OBS')

#%%
test_data['BMI_OBS'].value_counts()

#%%
print_value_labels(meta, 'Alcohol_8wPP')

#%%
import numpy as np

#%%
# replace 0.0 by 1.0 in Problemspreg_8wPP
test_data['Problemspreg_8wPP'] = test_data['Problemspreg_8wPP'].replace(0.0, 1.0)

#%%
# replace empty string by NaN in Season_birth_8wPP
test_data['Season_birth_8wPP'] = test_data['Season_birth_8wPP'].replace('', np.nan)

#%%
# replace 9.0 by NaN in Gestart_medicatie_tijdens_zorg_OBS
test_data['Gestart_medicatie_tijdens_zorg_OBS'] = test_data['Gestart_medicatie_tijdens_zorg_OBS'].replace(9.0, np.nan)	

#%%
# replace 9.0 by NaN in Preexistente_aandoeningen_OBS
test_data['Preexistente_aandoeningen_OBS'] = test_data['Preexistente_aandoeningen_OBS'].replace(9.0, np.nan)	

#%%
# replace 9.0 by NaN in Verdenking_intra_uteriene_groeivertraging_OBS
test_data['Verdenking_intra_uteriene_groeivertraging_OBS'] = test_data['Verdenking_intra_uteriene_groeivertraging_OBS'].replace(9.0, np.nan)

#%%
# replace empty strings with NaN in HC_mm_OBS
test_data['HC_mm_OBS'] = test_data['HC_mm_OBS'].replace('', np.nan)

#%%
# replace 4.0 and 2.0 by NaN in Type_partus_OBS
test_data['Type_partus_OBS'] = test_data['Type_partus_OBS'].replace([4.0, 2.0], np.nan)

#%%
# replace 3.0 by NaN in Partus_locatie_OBS
test_data['Partus_locatie_OBS'] = test_data['Partus_locatie_OBS'].replace(3.0, np.nan)

#%%
# replace 33.0 by 3.0 in Breken_van_de_vliezen_OBS
test_data['Breken_van_de_vliezen_OBS'] = test_data['Breken_van_de_vliezen_OBS'].replace(33.0, 3.0)

# replace 99.0 and 2.0 by NaN in Breken_van_de_vliezen_OBS
test_data['Breken_van_de_vliezen_OBS'] = test_data['Breken_van_de_vliezen_OBS'].replace([99.0, 2.0], np.nan)

#%%
# replace 99.0 and 9.0 by NaN in Pijnstilling_type_OBS
test_data['Pijnstilling_type_OBS'] = test_data['Pijnstilling_type_OBS'].replace([99.0, 9.0], np.nan)

# replace 2.0 by 3.0 in Pijnstilling_type_OBS
test_data['Pijnstilling_type_OBS'] = test_data['Pijnstilling_type_OBS'].replace(2.0, 3.0)

#%%
# replace 0.0 by NaN in Ligging_kind_OBS
test_data['Ligging_kind_OBS'] = test_data['Ligging_kind_OBS'].replace(0.0, np.nan)

#%%
# replace 99.0 and 8.0 by NaN in Conditie_perineum_OBS
test_data['Conditie_perineum_OBS'] = test_data['Conditie_perineum_OBS'].replace([99.0, 8.0], np.nan)

#%%
# replace 4.0, 2.0 and 3.0 by NaN in Complicaties_na_bevalling_OBS
test_data['Complicaties_na_bevalling_OBS'] = test_data['Complicaties_na_bevalling_OBS'].replace([4.0, 2.0, 3.0], np.nan)

#%%
# replace 2.0 by NaN in Kind_geslacht_OBS
test_data['Kind_geslacht_OBS'] = test_data['Kind_geslacht_OBS'].replace(2.0, np.nan)

#%%
# replace 999.0 by NaN in Kind_apgar_1_minuut_OBS
test_data['Kind_apgar_1_minuut_OBS'] = test_data['Kind_apgar_1_minuut_OBS'].replace(999.0, np.nan)

#%%
# replace 999.0 by NaN in Kind_apgar_5_minuten_OBS
test_data['Kind_apgar_5_minuten_OBS'] = test_data['Kind_apgar_5_minuten_OBS'].replace(999.0, np.nan)

#%%
# replace 999.0 by NaN in Kind_apgar_10_minuten_OBS
test_data['Kind_apgar_10_minuten_OBS'] = test_data['Kind_apgar_10_minuten_OBS'].replace(999.0, np.nan)

#%%
# replace 9.0 by NaN in Kind_voeding_OBS
test_data['Kind_voeding_OBS'] = test_data['Kind_voeding_OBS'].replace(9.0, np.nan)

# replace 10.0 by 1.0 in Kind_voeding_OBS
test_data['Kind_voeding_OBS'] = test_data['Kind_voeding_OBS'].replace(10.0, 1.0)

#%%
# replace 9.0 by NaN in Consult_kinderarts_OBS
test_data['Consult_kinderarts_OBS'] = test_data['Consult_kinderarts_OBS'].replace(9.0, np.nan)

#%%
# create BMI_OBS using 'Start_gewicht_OBS' and 'Lengte_OBS'
test_data['BMI_OBS'] = test_data['Start_gewicht_OBS'] / (test_data['Lengte_OBS'] / 100) ** 2

#%%
columns_to_keep = ['Graviditeit_OBS', 'Pariteit_OBS', 'Mater_voor_partus_OBS', 'OVG_vroeggeboorte_OBS',
                   'BMI_OBS', 'Begin_medicatiegebruik_OBS', 'Gestart_medicatie_tijdens_zorg_OBS',
                   'Preexistente_aandoeningen_OBS', 'Complicaties_zwangerschap_OBS',
                   'Verwijzing_zwangerschap_OBS', 'AC_mm_OBS', 'TCD_mm_OBS', 'FL_mm_OBS',
                   'Baring_amenorroeduur_weken_OBS', 'Baring_amenorroeduur_dagen_OBS',
                   'Para_voor_huidige_partus_OBS', 'Partus_locatie_OBS', 'Type_partus_OBS',
                   'Breken_van_de_vliezen_OBS', 'Baringsuitkomst_OBS', 'Ligging_kind_OBS',
                   'Meconium_OBS', 'Fluxus_OBS', 'Aantal_navelstrengvaten_OBS', 'Kind_geslacht_OBS',
                   'Kind_gewicht_gram_OBS', 'Kind_apgar_1_minuut_OBS', 'Kind_apgar_5_minuten_OBS',
                   'Kind_apgar_10_minuten_OBS', 'Kind_voeding_OBS', 'Consult_kinderarts_OBS',
                   'Practice_OBS', 'BMI_Start_gewicht_OBS', 'SGA_OBS', 'LGA_OBS', 'AGA_OBS',
                   'Zwangerschapshypertensie_OBS']

columns_cat = ['Partus_locatie_OBS', 'Breken_van_de_vliezen_OBS', 'Baringsuitkomst_OBS',
               'Ligging_kind_OBS', 'Meconium_OBS', 'Practice_OBS']