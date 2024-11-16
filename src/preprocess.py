# !/usr/bin/python
import pandas as pd
import warnings
import util

warnings.filterwarnings('ignore')

class data_preprocess(object):
    def read_data(self, path):
        print("!!! Data loading !!!")
        patient_df = pd.read_json(path, orient='columns')

        patient_df['id'] = patient_df['patient']['id']
        patient_df['gender'] = patient_df['patient']['gender']
        patient_df['dateOfBirth'] = patient_df['patient']['dateOfBirth']
        patient_df['race'] = patient_df['patient']['race']
        patient_df['diagnosis'] = patient_df['patient']['diagnosis']

        for value, key in patient_df['patient']['address'].items():
            patient_df[value] = key

        for value, key in patient_df['aiDataset']['data'][0].items():
            if value == "date":
                patient_df[value] = key
            else:
                for value, key in patient_df['aiDataset']['data'][0]['socialScore'].items():
                    if(value not in ('lastThreeMonths','lastSixMonths','lastTwelveMonths')):
                        patient_df[value] = key

        for month in ('lastThreeMonths','lastSixMonths','lastTwelveMonths'):
            for value, key in patient_df['aiDataset']['data'][0]['socialScore'][month][0].items():
                value = month + "_" + value
                patient_df[value] = key

        patient_df = patient_df.reset_index(drop=True)
        patient_df = patient_df.drop(columns=['patient', 'aiDataset'])
        patient_df = patient_df.drop_duplicates()
        patient_df.shape

        patient_df['dateOfBirth'] = pd.to_datetime(patient_df['dateOfBirth'], yearfirst = True)

        patient_df['age'] = util.age(patient_df['dateOfBirth'])
