#!/usr/bin/python
import psycopg2
from config import config
import pandas.io.sql as psql
from datetime import datetime, date

import os
import pandas as pd
import numpy as np


def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


def calc_oud_score(patient_df):
    print("calc_oud_score  process started")

    def calc_age_weight(age):
        if (age > 65):
            age_weight = 1
        elif (age <= 65 and age > 44):
            age_weight = 3
        elif (age <= 44):
            age_weight = 5
        else:
            age_weight = 0
        return age_weight

    def calc_gender_weight(gender):
        if (gender == 'female'):
            gender_weight = 1
        elif (gender == 'male'):
            gender_weight = 3
        else:
            gender_weight = 5
        return gender_weight

    def calc_race_weight(race):
        if (race == 'white'):
            race_weight = 1
        elif (race == 'other'):
            race_weight = 3
        elif (race == 'black'):
            race_weight = 5
        else:
            race_weight = 5
        return race_weight

    def calc_bmi_weight(bmi):
        if (bmi < 24.9):
            bmi_weight = 1
        elif (age >= 25 and age < 30):
            bmi_weight = 3
        elif (age >= 30 and age < 35):
            bmi_weight = 5
        else:
            bmi_weight = 10
        return bmi_weight

    def calc_timezone_weight(timezone):
        if (timezone == 'West/North-East'):
            timezone_weight = 1
        elif (timezone == 'Mid-West'):
            timezone_weight = 3
        elif (timezone == 'South'):
            timezone_weight = 5
        else:
            timezone_weight = 10
        return timezone_weight

    def calc_socioeconomic_weight(socioEconomic):
        if (socioEconomic == 'None'):
            socioeconomic_weight = 0
        elif (socioEconomic == 'Upper'):
            socioeconomic_weight = 1
        elif (socioEconomic == 'Middle'):
            socioeconomic_weight = 3
        elif (socioEconomic == 'Lower'):
            socioeconomic_weight = 5
        else:
            socioeconomic_weight = 5
        return socioeconomic_weight

    def calc_alcohol_weight(alcohol):
        if (alcohol == 'None'):
            alcohol_weight = 0
        elif (alcohol == 'Former'):
            alcohol_weight = 3
        elif (alcohol == 'Current'):
            alcohol_weight = 5
        elif (alcohol == 'Frequent'):
            alcohol_weight = 10
        else:
            alcohol_weight = 10
        return alcohol_weight

    def calc_drug_use_weight(drugUse):
        if (drugUse == 'None'):
            drug_use_weight = 0
        elif (drugUse == 'Former'):
            drug_use_weight = 3
        elif (drugUse == 'Current'):
            drug_use_weight = 5
        elif (drugUse == 'Frequent'):
            drug_use_weight = 10
        else:
            drug_use_weight = 10
        return drug_use_weight

    def calc_smoking_status_weight(smokingStatus):
        if (smokingStatus == 'None'):
            smoking_status_weight = 0
        elif (smokingStatus == 'Former'):
            smoking_status_weight = 3
        elif (smokingStatus == 'Current'):
            smoking_status_weight = 5
        else:
            smoking_status_weight = 10
        return smoking_status_weight

    def type_constant(feature):
        if (feature == 'age'):
            feature_cons = 1
        elif (feature == 'gender'):
            feature_cons = 2
        elif (feature == 'race'):
            feature_cons = 1
        elif (feature == 'bmi'):
            feature_cons = 2
        elif (feature == 'timezone'):
            feature_cons = 1
        elif (feature == 'socioEconomic'):
            feature_cons = 1
        elif (feature == 'alcohol'):
            feature_cons = 1
        elif (feature == 'drugUse'):
            feature_cons = 2
        elif (feature == 'smokingStatus'):
            feature_cons = 2
        else:
            feature_cons = 0
        return feature_cons

    def oud_risk_score_calculator(age, gender, race, bmi, timezone, socioEconomic, alcohol, drugUse, smokingStatus):
        print("oud_risk_score_calculator  process started")

        age_score = calc_age_weight(age) * type_constant('age')
        gender_score = calc_gender_weight(gender) * type_constant('gender')
        race_score = calc_race_weight(race) * type_constant('race')
        bmi_score = calc_bmi_weight(bmi) * type_constant('bmi')
        timezone_score = calc_timezone_weight(timezone) * type_constant('timezone')
        socioEconomic_score = calc_socioeconomic_weight(socioEconomic) * type_constant('socioEconomic')
        alcohol_score = calc_alcohol_weight(alcohol) * type_constant('alcohol')
        drugUse_score = calc_drug_use_weight(drugUse) * type_constant('drugUse')
        smokingStatus_score = calc_smoking_status_weight(smokingStatus) * type_constant('smokingStatus')

        calculator_score = age_score + gender_score + race_score + bmi_score + timezone_score + socioEconomic_score + alcohol_score + drugUse_score + smokingStatus_score
        return calculator_score

    def scale_cal_score(score):
        print("scale_cal_score  process started")

        score = ((100 * (score - 7)) / 98) / 100
        # score = score.apply(lambda x: '{:.5f}'.format(x))
        return round(score, 5)

    age = patient_df['age']
    gender = patient_df['gender']
    race = patient_df['race']
    bmi = patient_df['bmi']
    timezone = 'other'
    socioEconomic = patient_df['socioEconomic']
    alcohol = patient_df['alcohol']
    drugUse = patient_df['drugUse']
    smokingStatus = patient_df['smokingStatus']
    score = oud_risk_score_calculator(age, gender, race, bmi, timezone, socioEconomic, alcohol, drugUse, smokingStatus)

    scaled_score = scale_cal_score(score)

    return scaled_score


def age(birth_date):
    dates = datetime.strptime(birth_date, '%Y-%m-%d')
    today = date.today()
    age = today.year - dates.year
    return age

def risk_calculator(score):
    print("risk_calculator  process started")
    if score > 0.7:#.astype(float) > 0.7:
        score = 'High'
    elif score < 0.5:#.astype(float) < 0.5:
        score = 'Low'
    else:
        score = 'Medium'

    return score


def calculate_opioid_score(final_df):
    print("calculate_opioid_score  process started")

    final_df['OUD_Score'] = (final_df['Random_Forest_Probability'] * 0.90) + (
                final_df['Calibrated_Random_Forest_Probability'] * 0.90) + (
                                        final_df['Naive_Bias_Probability'] * 0.69) + (
                                        final_df['Isotonic_Calibrated_Naive_Bias_Probability'] * 0.69) + (
                                        final_df['Sigmoid_Calibrated_Naive_Bias_Probability'] * 0.68)

    def NormalizeData(data):
        print("NormalizeData  process started")
        return (data - np.min(data)) / (np.max(data) - np.min(data))

    final_df['OUD_Score'] = NormalizeData(final_df['OUD_Score'])
    return final_df

def read_postgre_data():
    """ Connect to the PostgreSQL database server """
    conn = None
    
    try:
        # read connection parameters
        params = config()

        # connect to the PostgreSQL server
        print('Connecting to the PostgreSQL database...')
        conn = psycopg2.connect(**params)

        # create a cursor
        cur = conn.cursor()
        
        # Open and read the file as a single buffer
        sql_path = os.path.join("sql", "model_data.sql")
        with open(sql_path) as query_string:
            postgreSQL_select_Query = query_string.read()
        
        dataframe = psql.read_sql(postgreSQL_select_Query, conn)
        print("********", type(dataframe))
        return dataframe

    # close the communication with the PostgreSQL
        cur.close()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()
            print('Database connection closed.')


def write_data_postgre(records):
    """ Connect to the PostgreSQL database server """

    conn = None
    try:
        # read connection parameters
        params = config()
        # connect to the PostgreSQL server
        print('Connecting to the PostgreSQL database...')
        conn = psycopg2.connect(**params)

        # create a cursor
        cur = conn.cursor()

        for index, row in records.iterrows():
            patient_id = row["patient_id"]
            Opioid_Score = row["Opioid_Score"]
            Opioid_Risk = row["Opioid_Risk"]
            Prediction_Source = row["Prediction_Source"]
            Created_On = row["Created_On"]
   
            insert_sql = f""" INSERT INTO postgres.public."predictionOutputs" ("patient_id", "Opioid_Score", "Opioid_Risk", "Prediction_Source", "Created_On") VALUES ('{patient_id}','{Opioid_Score}','{Opioid_Risk}','{Prediction_Source}', '{Created_On}');"""
            cur.execute(insert_sql)

        # Open and read the file as a single buffer
        update_sql = os.path.join("sql", "update_flag.sql")
        with open(update_sql) as update_query_string:
            postgreSQL_update_Query = update_query_string.read()

        #Update Has_Predicted column in Model_Data
        #update_sql = """ UPDATE public."modelData" SET "hasPredicted" = true where "hasPredicted" = false; """

        cur.execute(postgreSQL_update_Query)
        conn.commit()
        conn.close()

    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()
            print('Database connection closed.')