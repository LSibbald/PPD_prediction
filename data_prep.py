import pandas as pd
import pyreadstat
import numpy as np

class DataCleaner:
    def __init__(self, filepath):
        self.filepath = filepath 
        self.data, self.meta = pyreadstat.read_sav(filepath)
    
    def update_variable_labels(self, variable_name, new_labels):
        if variable_name in self.meta.variable_value_labels:
            current_labels = self.meta.variable_value_labels[variable_name]
            updated_labels = {}
            for old_key, new_key in new_labels.items():
                if old_key in current_labels:
                    updated_labels[new_key] = current_labels[old_key]
            self.meta.variable_value_labels[variable_name] = updated_labels
        else:
            print(f"Variable '{variable_name}' not found in variable_value_labels.")

    def prepare_week12_data(self):
        # Define new labels mapping for a variable, e.g., SMUtime_12
        new_labels_mapping_SMUtime_12 = {
            1: 0,
            2: 0.25,
            3: 0.5,
            4: 1,
            5: 2,
            6: 3,
            7: 4,
            8: 5,
            9: 6,
            10: 7
        }

        new_labels_mapping_SMUday_week_12 = {	
            1: 0,	
            2: 1,	
            3: 2,	
            4: 3,	
            5: 4,	
            6: 5,	
            7: 6,
            8: 7	
        }

        new_labels_mapping_nausea_12 = {
            1: 0,
            2: 1,
            3: 2,
            4: 3,
            5: 4
        }

        # Update the variable labels in the metadata
        self.update_variable_labels('SMUtime_12', new_labels_mapping_SMUtime_12)
        self.update_variable_labels('SMUday_week_12', new_labels_mapping_SMUday_week_12)
        self.update_variable_labels('Nausea_12', new_labels_mapping_nausea_12)

        # Data cleaning steps
        self.data['UserLanguage_12'] = self.data['UserLanguage_12'].replace(['EN', 'NL'], [0, 1])

        week12_columns = [col for col in self.data.columns if '_12' in col]

        # Add 'EPDS_TOT_8wPP'and 'RecipientFirstName' to the list
        week12_columns.extend(['EPDS_TOT_8wPP','RecipientFirstName'])
        
        # Subset the DataFrame
        week12_data = self.data[week12_columns]

        # Rename 'RecipientFirstName' to 'Participant'
        week12_data = week12_data.rename(columns={'RecipientFirstName': 'Participant'})

        # Reset index
        week12_data = week12_data.set_index('Participant')

        # Subset the week 12 columns
        columns_to_keep = ["UserLanguage_12", "Age_12", "Maritalstatus_12", "Genderpartner_12", "Education_12",
                    "Population_12", "Sexdetermination_12", "Sexpreference_12", "Pregnancyloss_12",
                    "Prevdelivery_12", "Plannedpreg_12",
                    "Locationdelivery_12", "Problemspreg_12", "Painmanagement_12",
                    "Nausea_12", "Nausea_weeks_12", "Work_12", "Unpaidwork_12", "Worksituation_12", "Workpostpreg_12",
                    "Workpartner_12", "Hoursworkpartner_12", "Breadwinner_12", "Absence_illness_12",
                    "Alcohol_12", "Smoking_12", "Smokingpartner_12", "Lifeevent_12", "Thyroid_mother_12",
                    "Thyroid_father_12", "Thyroid_aunt_12", "Thyroid_uncle_12", "Thyroid_sibling_12",
                    "Hypertension_12", "Eczema_12", "Rheumatism_12", "Diabetes_12", "Chronicdisease_other_12",
                    "BPmother_12", "Diabetesmother_12", "CVDmother_12", "GDMmother_12", "BPpregmother_12",
                    "Historydepression_12", "Historyanxiety_12", "Treatment_12", "Supplements_12", "Medication_12",
                    "Medication_unreg_12", "SMUtime_12", "Complaints_2_12", "Complaints_4_12", "Complaints_6_12",
                    "Complaints_8_12", "Complaints_10_12", "Complaints_11_12", "Complaints_13_12", "Complaints_17_12",
                    "SMUday_week_12", "SMUmotives_1_12", "SMUmotives_2_12", "SMUmotives_3_12",
                    "SMUmotives_4_12", "SMUmotives_5_12", "SMUmotives_6_12", "SMUmotives_7_12", "SMUmotives_8_12",
                    "SMUmotives_9_12", "SMUmotives_10_12", "SMUmotives_11_12", "SMUmotives_12_12", "SMUmotives_13_12",
                    "SMUmotives_14_12", "SMUmotives_15_12", "SMUmotives_16_12", "SMUmotives_17_12", "SMUmotives_18_12",
                    "TSHH_12", "FT4H_12", "HCGBH_12", "ATPOH_12", "AMHH_12", "ACOVH_12", "ACOVKH_12", "CRP4H_12",
                    "FERRH_12", "IL6H_12", "NBNPH_12", "PLGFH_12", "SFLTH_12", "TRFE_12", "BRA1_12", "BRA2_12",
                    "BRABF_12", "BRA4_12", "TPDS_TOT_12", "EDS_TOT_12", "SCL90_TOT_12", "Scale_forgetfulness_12",
                    "Scale_concentrating_12", "Sleep_12", "BMI_12", "DAS_Satisfaction_TOT_12", "PUQE_TOT_12",
                    "Work_Engagement_TOT_12", "Work_Burnout_TOT_12", "VBBA_Supervisor_TOT_12",
                    "VBBA_Colleagues_TOT_12", "IWPQ_TP_TOT_12", "IWPQ_CP_TOT_12", "DS14_NA_TOT_12",
                    "DS14_SI_TOT_12", "TypeD_12", "BSMAS_TOT_12", "EPDS_TOT_8wPP"]

        week12_subset = week12_data[columns_to_keep]

        columns_cat = ["Maritalstatus_12", "Education_12", "Population_12", "Sexdetermination_12", "Sexpreference_12",
                "Locationdelivery_12", "Painmanagement_12", "Work_12",
                "Worksituation_12", "Workpostpreg_12", "Workpartner_12", "Hoursworkpartner_12", "Breadwinner_12",
                "Alcohol_12", "Smokingpartner_12", "BPmother_12", "Diabetesmother_12", "CVDmother_12",
                "GDMmother_12", "BPpregmother_12", "Treatment_12", "Nausea_weeks_12"]
        
        for col in columns_cat:
            if col in self.meta.variable_value_labels:
                week12_subset[col] = week12_subset[col].replace(self.meta.variable_value_labels[col])

        # Make sure all columns are numeric
        week12_subset['UserLanguage_12'] = pd.to_numeric(week12_subset['UserLanguage_12'], errors='coerce')
        
        return week12_subset
    
    def prepare_week20_data(self):
        # Generate week 12 data using the existing function
        week12_data = self.prepare_week12_data()

        # Define new labels mapping for a variable, e.g., SMUtime_20
        new_labels_mapping_SMUtime_20 = {
            1: 0,
            2: 0.25,
            3: 0.5,
            4: 1,
            5: 2,
            6: 3,
            7: 4,
            8: 5,
            9: 6,
            10: 7
        }

        new_labels_mapping_nausea_20 = {
            1: 0,
            2: 1,
            3: 2,
            4: 3,
            5: 4
        }

        # Update the variable labels in the metadata
        self.update_variable_labels('SMUtime_20', new_labels_mapping_SMUtime_20)
        self.update_variable_labels('Nausea_20', new_labels_mapping_nausea_20)

        # Assuming the variable names for perfectionism calculation
        perfectionism_vars = [
            'Perfectionism_push_20', 'Perfectionism_standards_20', 'Perfectionism_failure_20',
            'Perfectionism_judge_20', 'Perfectionism_striving_20', 'Perfectionism_careful_20',
            'Perfectionism_ambition_20'
        ]

        # Add the new perfectionism_20 variable as the sum of the specified variables
        self.data['perfectionism_20'] = self.data[perfectionism_vars].sum(axis=1)

        # Subset the week 20 columns
        week20_columns = [col for col in self.data.columns if '_20' in col]
        
        # Add 'RecipientFirstName' to the list
        week20_columns.extend(['RecipientFirstName'])
        
        # Subset the DataFrame
        week20_data = self.data[week20_columns]

        # Rename 'RecipientFirstName' to 'Participant'
        week20_data = week20_data.rename(columns={'RecipientFirstName': 'Participant'})

        # Reset index
        week20_data = week20_data.set_index('Participant')

        # Subset the week 20 columns
        columns_to_keep = ['Pregguidance_20', 'Gynaecologist_20', 'Problemspreg_20', 'Particularityecho_20',
                        'Healthinsurance_20', 'Partner_20', 'Alcohol_20',
                        'Smoking_20', 'Smokingpartner_20', 'Doctorvisit_20', 'Medication_20',
                        'Supplements_20', 'Lifeevent_20', 'Nausea_20', 'Nausea_weeks_20', 'SMUtime_20',
                        'Complaints_2_20', 'Complaints_4_20', 'Complaints_6_20', 'Complaints_8_20',
                        'Complaints_10_20', 'Complaints_11_20','Complaints_13_20', 'Complaints_17_20',
                        'Medication_unreg_20', 'TSHH_20', 'FT4H_20', 'HCGBH_20', 'ATPOH_20',
                        'ACOVH_20', 'ACOVKH_20', 'CRP4H_20', 'FERRH_20', 'IL6H_20', 'NBNPH_20',
                        'PLGFH_20', 'SFLTH_20', 'BRABF_20', 'TPDS_TOT_20', 'EDS_TOT_20',
                        'SCL90_TOT_20', 'Scale_forgetfulness_20','Scale_concentrating_20',
                        'Sleep_20', 'BMI_20', 'DAS_Satisfaction_TOT_20', 'MSPSS_Family_TOT_20',
                        'MSPSS_Friends_TOT_20', 'TFMQ_SF_Nonreacting_TOT_20',
                        'TFMQ_SF_Nonjudging_TOT_20', 'TFMQ_SF_Awareness_TOT_20', 'PUQE_TOT_20',
                        'DERS16_Clarity_TOT_20', 'DERS16_Goals_TOT_20', 'DERS16_Strategies_TOT_20',
                        'DERS16_Nonacceptance_TOT_20', 'BSMAS_TOT_20', 'perfectionism_20']
        
        week20_subset = week20_data[columns_to_keep]
        
        columns_cat = ['Pregguidance_20', 'Gynaecologist_20', 'Particularityecho_20', 'Healthinsurance_20',
               'Alcohol_20', 'Smokingpartner_20', 'Nausea_weeks_20']
        
        for col in columns_cat:
            if col in self.meta.variable_value_labels:
                week20_subset[col] = week20_subset[col].replace(self.meta.variable_value_labels[col])

        # Merge the week 20 data with week 12 data
        merged_data = week20_subset.join(week12_data, how='inner')

        # Return the merged dataset
        return merged_data
    
    def prepare_week28_data(self):
        # Generate week 20 data using the existing function
        week20_data = self.prepare_week20_data()

        # Define new labels mapping for a variable, e.g., SMUtime_28
        new_labels_mapping_SMUtime_28 = {
            1: 0,
            2: 0.25,
            3: 0.5,
            4: 1,
            5: 2,
            6: 3,
            7: 4,
            8: 5,
            9: 6,
            10: 7
        }

        new_labels_mapping_nausea_28 = {
            1: 0,
            2: 1,
            3: 2,
            4: 3,
            5: 4
        }

        # Update the variable labels in the metadata
        self.update_variable_labels('SMUtime_28', new_labels_mapping_SMUtime_28)
        self.update_variable_labels('Nausea_28', new_labels_mapping_nausea_28)

        # Subset the week 28 columns
        week28_columns = [col for col in self.data.columns if '_28' in col]
        
        # Add 'RecipientFirstName' to the list
        week28_columns.extend(['RecipientFirstName'])
        
        # Subset the DataFrame
        week28_data = self.data[week28_columns]

        # Rename 'RecipientFirstName' to 'Participant'
        week28_data = week28_data.rename(columns={'RecipientFirstName': 'Participant'})

        # Reset index
        week28_data = week28_data.set_index('Participant')

        # Subset the week 28 columns
        columns_to_keep = ['Pregguidance_28', 'Gynaecologist_28', 'Problemspreg_28','Alcohol_28', 'Smoking_28',
                           'Lifeevent_28', 'Doctorvisit_28', 'Supplements_28', 'Medication_28', 
                           'Medication_unreg_28', 'Partner_28', 'Nausea_28', 'Nausea_weeks_28',
                           'SMUtime_28', 'Work_28', 'Scale_forgetfulness_28', 'Scale_concentrating_28',
                           'Complaints_2_28', 'Complaints_4_28', 'Complaints_6_28', 'Complaints_8_28',
                           'Complaints_10_28', 'Complaints_11_28','Complaints_13_28', 'Complaints_17_28',
                           'TSHH_28', 'FT4H_28', 'HCGBH_28', 'ATPOH_28', 'ACOVH_28', 'ACOVKH_28',
                           'CRP4H_28', 'FERRH_28', 'IL6H_28', 'NBNPH_28', 'PLGFH_28', 'SFLTH_28',
                           'BRABF_28', 'TPDS_TOT_28', 'EDS_TOT_28', 'SCL90_TOT_28', 'Cognitive_TOT_28',
                           'Sleep_28', 'BMI_28', 'DAS_Satisfaction_TOT_28', 'BFI_2S_Extraversion_TOT_28',
                           'BFI_2S_Openmindedness_TOT_28', 'BFI_2S_Agreeableness_TOT_28',
                           'BFI_2S_Conscientiousness_TOT_28', 'BFI_2S_NE_TOT_28', 'PUQE_TOT_28',
                           'Work_Engagement_TOT_28', 'Work_Burnout_TOT_28', 'VBBA_Supervisor_TOT_28',
                           'VBBA_Colleagues_TOT_28', 'PPBS_TOT_28', 'IWPQ_TP_TOT_28', 'IWPQ_CP_TOT_28',
                           'BSMAS_TOT_28', 'Smokingpartner_28']
        
        week28_subset = week28_data[columns_to_keep]

        columns_cat =['Pregguidance_28', 'Gynaecologist_28', 'Nausea_weeks_28', 'Work_28', 'Alcohol_28',
                      'Smokingpartner_28']

        for col in columns_cat:
            if col in self.meta.variable_value_labels:
                week28_subset[col] = week28_subset[col].replace(self.meta.variable_value_labels[col])

        # Merge the week 28 data with week 20 data
        merged_data = week28_subset.join(week20_data, how='inner')

        # Return the merged dataset
        return merged_data
    
    def prepare_postpartum_data(self):
        # Generate week 28 data using the existing function
        week28_data = self.prepare_week28_data()

        # Define new labels mapping for a variable and replace values
        # Replace 0.0 by 1.0 in Problemspreg_8wPP
        self.data['Problemspreg_8wPP'] = self.data['Problemspreg_8wPP'].replace(0.0, 1.0)

        # Replace 0.0 by NaN in Opinioncrying_8wPP
        self.data['Opinioncrying_8wPP'] = self.data['Opinioncrying_8wPP'].replace(0.0, np.nan)

        # Replace 0.0 by NaN in Defecationbaby_8wPP
        self.data['Defecationbaby_8wPP'] = self.data['Defecationbaby_8wPP'].replace(0.0, np.nan)	

        new_labels_mapping_Defecationbaby_8wPP = {
            1: 0,
            2: 1,
            3: 2,
            4: 3,
            5: 4
        }

        # Replace 0.0 by 1.0 in Doctorvisitlastweeks_8wPP
        self.data['Doctorvisitlastweeks_8wPP'] = self.data['Doctorvisitlastweeks_8wPP'].replace(0.0, 1.0)

        new_labels_mapping_Doctorvisitlastweeks_8wPP = {
            1: 0,
            2: 1
        }

        # Replace empty string by NaN in Season_birth_8wPP
        self.data['Season_birth_8wPP'] = self.data['Season_birth_8wPP'].replace('', np.nan)

        # Update the variable labels in the metadata
        self.update_variable_labels('Defecationbaby_8wPP', new_labels_mapping_Defecationbaby_8wPP)
        self.update_variable_labels('Doctorvisitlastweeks_8wPP', new_labels_mapping_Doctorvisitlastweeks_8wPP)

        # Subset the 8wPP columns
        PP_columns = [col for col in self.data.columns if '_8wPP' in col]
        
        # Add 'RecipientFirstName' to the list
        PP_columns.extend(['RecipientFirstName'])
        
        # Subset the DataFrame
        PP_data = self.data[PP_columns]

        # Rename 'RecipientFirstName' to 'Participant'
        PP_data = PP_data.rename(columns={'RecipientFirstName': 'Participant'})

        # Reset index
        PP_data = PP_data.set_index('Participant')

        columns_to_keep= ['Pregguidance_8wPP', 'Gynaecologist_8wPP', 'Problemspreg_8wPP', 'Pregduration_8wPP',
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
                   'DERS16_Strategies_TOT_8wPP', 'DERS16_Nonacceptance_TOT_8wPP', 'GAD_7_TOT_8wPP',
                   'OBVL_Parent_Child_Relationship_8wPP', 'OBVL_Parenting_8wPP', 'OBVL_Role_Limitation_8wPP']
        
        PP_subset = PP_data[columns_to_keep]

        columns_cat = ['Pregguidance_8wPP', 'Gynaecologist_8wPP', 'Problemspreg_8wPP', 'Delivery_8wPP',
                'Spinalpuncture_8wPP', 'Checkup_8wPP', 'Checkupresults_8wPP', 'Nutrition_8wPP',
                'Workpostpreg_8wPP', 'Workchanges_8wPP', 'Workpartner_8wPP', 'Hoursjobpartner_8wPP',
                'Breadwinner_8wPP',  'Opinioncrying_8wPP', 'Alcohol_8wPP',
                'Smokingpartner_8wPP']
        
        for col in columns_cat:
            if col in self.meta.variable_value_labels:
                PP_subset[col] = PP_subset[col].replace(self.meta.variable_value_labels[col])

        # add Season_birth_8wPP to the list
        columns_cat.extend(['Season_birth_8wPP'])

        # Convert all values in columns_cat to strings
        PP_subset[columns_cat] = PP_subset[columns_cat].astype(str)

        # Merge the week 28 data with pp data
        merged_data = PP_subset.join(week28_data, how='inner')

        # Return the merged dataset
        return merged_data
    
    def prepare_obs_data(self):
        # Generate 8wPP data using the existing function
        PP_data = self.prepare_postpartum_data()

        # Replace value labels when necessary
        # Replace 9.0 by NaN in Gestart_medicatie_tijdens_zorg_OBS
        self.data['Gestart_medicatie_tijdens_zorg_OBS'] = self.data['Gestart_medicatie_tijdens_zorg_OBS'].replace(9.0, np.nan)

        # Replace 9.0 by NaN in Preexistente_aandoeningen_OBS
        self.data['Preexistente_aandoeningen_OBS'] = self.data['Preexistente_aandoeningen_OBS'].replace(9.0, np.nan)

        # Replace 9.0 by NaN in Verdenking_intra_uteriene_groeivertraging_OBS
        self.data['Verdenking_intra_uteriene_groeivertraging_OBS'] = self.data['Verdenking_intra_uteriene_groeivertraging_OBS'].replace(9.0, np.nan)

        # Replace empty strings with NaN in HC_mm_OBS
        self.data['HC_mm_OBS'] = self.data['HC_mm_OBS'].replace('', np.nan)

        # Replace 4.0 and 2.0 by NaN in Type_partus_OBS
        self.data['Type_partus_OBS'] = self.data['Type_partus_OBS'].replace([4.0, 2.0], np.nan)

        # Replace 3.0 by NaN in Partus_locatie_OBS
        self.data['Partus_locatie_OBS'] = self.data['Partus_locatie_OBS'].replace(3.0, np.nan)

        # Replace 33.0 by 3.0 in Breken_van_de_vliezen_OBS
        self.data['Breken_van_de_vliezen_OBS'] = self.data['Breken_van_de_vliezen_OBS'].replace(33.0, 3.0)

        # Replace 99.0 and 2.0 by NaN in Breken_van_de_vliezen_OBS
        self.data['Breken_van_de_vliezen_OBS'] = self.data['Breken_van_de_vliezen_OBS'].replace([99.0, 2.0], np.nan)

        # Replace 99.0 and 9.0 by NaN in Pijnstilling_type_OBS
        self.data['Pijnstilling_type_OBS'] = self.data['Pijnstilling_type_OBS'].replace([99.0, 9.0], np.nan)

        # Replace 2.0 by 3.0 in Pijnstilling_type_OBS
        self.data['Pijnstilling_type_OBS'] = self.data['Pijnstilling_type_OBS'].replace(2.0, 3.0)

        # Replace 0.0 by NaN in Ligging_kind_OBS
        self.data['Ligging_kind_OBS'] = self.data['Ligging_kind_OBS'].replace(0.0, np.nan)

        # Replace 99.0 and 8.0 by NaN in Conditie_perineum_OBS
        self.data['Conditie_perineum_OBS'] = self.data['Conditie_perineum_OBS'].replace([99.0, 8.0], np.nan)

        # Replace 4.0, 2.0 and 3.0 by NaN in Complicaties_na_bevalling_OBS
        self.data['Complicaties_na_bevalling_OBS'] = self.data['Complicaties_na_bevalling_OBS'].replace([4.0, 2.0, 3.0], np.nan)

        # Replace 2.0 by NaN in Kind_geslacht_OBS
        self.data['Kind_geslacht_OBS'] = self.data['Kind_geslacht_OBS'].replace(2.0, np.nan)

        # Replace 999.0 by NaN in Kind_apgar_1_minuut_OBS
        self.data['Kind_apgar_1_minuut_OBS'] = self.data['Kind_apgar_1_minuut_OBS'].replace(999.0, np.nan)

        # Replace 999.0 by NaN in Kind_apgar_5_minuten_OBS
        self.data['Kind_apgar_5_minuten_OBS'] = self.data['Kind_apgar_5_minuten_OBS'].replace(999.0, np.nan)

        # Replace 999.0 by NaN in Kind_apgar_10_minuten_OBS
        self.data['Kind_apgar_10_minuten_OBS'] = self.data['Kind_apgar_10_minuten_OBS'].replace(999.0, np.nan)

        # Replace 9.0 by NaN in Kind_voeding_OBS
        self.data['Kind_voeding_OBS'] = self.data['Kind_voeding_OBS'].replace(9.0, np.nan)

        # Replace 10.0 by 1.0 in Kind_voeding_OBS
        self.data['Kind_voeding_OBS'] = self.data['Kind_voeding_OBS'].replace(10.0, 1.0)

        # Replace 9.0 by NaN in Consult_kinderarts_OBS
        self.data['Consult_kinderarts_OBS'] = self.data['Consult_kinderarts_OBS'].replace(9.0, np.nan)

        # Subset the 8wPP columns
        OBS_columns = [col for col in self.data.columns if '_OBS' in col]

        # Add 'RecipientFirstName' to the list
        OBS_columns.extend(['RecipientFirstName'])
        
        # Subset the DataFrame
        OBS_data = self.data[OBS_columns]

        # Rename 'RecipientFirstName' to 'Participant'
        OBS_data = OBS_data.rename(columns={'RecipientFirstName': 'Participant'})

        # Reset index
        OBS_data = OBS_data.set_index('Participant')
        
        columns_to_keep = ['Graviditeit_OBS', 'Pariteit_OBS', 'Mater_voor_partus_OBS', 'OVG_vroeggeboorte_OBS',
                'Begin_medicatiegebruik_OBS', 'Gestart_medicatie_tijdens_zorg_OBS',
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
        
        OBS_subset = OBS_data[columns_to_keep]
        
        columns_cat = ['Partus_locatie_OBS', 'Breken_van_de_vliezen_OBS', 'Baringsuitkomst_OBS',
               'Ligging_kind_OBS', 'Meconium_OBS', 'Practice_OBS']
        
        for col in columns_cat:
            if col in self.meta.variable_value_labels:
                OBS_subset[col] = OBS_subset[col].replace(self.meta.variable_value_labels[col])

        # Merge the week 28 data with pp data
        merged_data = OBS_subset.join(PP_data, how='inner')

        # Return the merged dataset
        return merged_data