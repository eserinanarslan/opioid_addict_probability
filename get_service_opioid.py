import pandas as pd
import numpy as np
import os

import flask
from flask import request, jsonify
import configparser

import json
import warnings

warnings.filterwarnings("ignore")
pd.set_option('display.float_format', '{:.4f}'.format)

config = configparser.ConfigParser()
config.sections()
config.read('config.ini')
"""
conn = sql.connect(os.getcwd()+"/dataset/churn_df.db")
data_opioid.to_sql("churn_df", conn, if_exists='replace')

conn = sql.connect(os.getcwd()+"/dataset/oud_df.db")
conn2 = sql.connect(os.getcwd()+"/dataset/unique_oud_results.db")

data_opioid = pd.read_sql("SELECT * FROM oud_df", conn).drop(columns="index")
unique_data = pd.read_sql("SELECT * FROM unique_oud_df", conn2).drop(columns="index")
"""
data_opioid = pd.read_csv(os.getcwd() + "/dataset/oud_results.csv")
unique_data = pd.read_csv(os.getcwd() + "/dataset/unique_oud_results.csv")

data_opioid = data_opioid.rename({'OUD_Score': 'Opioid_Score', 'PatientId': 'patient_id'}, axis=1)
unique_data = unique_data.rename({'OUD_Score': 'Opioid_Score', 'PatientId': 'patient_id'}, axis=1)

data_opioid.fillna("NA", inplace=True)
unique_data.fillna("NA", inplace=True)


def column_format(data):
    data['Opioid_Risk'] = 'Medium'
    data['Random_Forest_Probability'] = data['Random_Forest_Probability'].apply(lambda x: '{:.5f}'.format(x))
    data['Calibrated_Random_Forest_Probability'] = data['Calibrated_Random_Forest_Probability'].apply(
        lambda x: '{:.5f}'.format(x))
    data['Naive_Bias_Probability'] = data['Naive_Bias_Probability'].apply(lambda x: '{:.5f}'.format(x))
    data['Isotonic_Calibrated_Naive_Bias_Probability'] = data['Isotonic_Calibrated_Naive_Bias_Probability'].apply(
        lambda x: '{:.5f}'.format(x))
    data['Sigmoid_Calibrated_Naive_Bias_Probability'] = data['Sigmoid_Calibrated_Naive_Bias_Probability'].apply(
        lambda x: '{:.5f}'.format(x))
    data['Opioid_Score'] = data['Opioid_Score'].apply(lambda x: '{:.5f}'.format(x))

    data.loc[data['Opioid_Score'].astype(float) > 0.7, 'Opioid_Risk'] = 'High'
    data.loc[data['Opioid_Score'].astype(float) < 0.5, 'Opioid_Risk'] = 'Low'
    data = data.reindex(columns=['patient_id', 'Opioid_Risk', 'Opioid_Score',
                                 'Calibrated_Random_Forest_Probability', 'Isotonic_Calibrated_Naive_Bias_Probability',
                                 'Naive_Bias_Probability',
                                 'Random_Forest_Probability', 'Sigmoid_Calibrated_Naive_Bias_Probability'])

    return data


data_opioid = column_format(data_opioid)
unique_data_opioid = column_format(unique_data)

df = data_opioid.to_json(orient="records")
df = json.loads(df)

df_unique = unique_data_opioid.to_json(orient="records")
df_unique = json.loads(df_unique)

app = flask.Flask(__name__)
app.config["DEBUG"] = True


@app.route('/patients_list', methods=['GET'])
def patients_list():

    patient_list = np.array(unique_data_opioid.patient_id.unique())
    json_str = json.dumps({'patient_list': patient_list.tolist()})
    return jsonify(json_str)


@app.route('/results', methods=['GET'])
def results():
    return jsonify(df[:100])


@app.route('/results/<patient_id>', methods=['GET'])
def api_id(patient_id):
    # Check if an ID was provided as part of the URL.
    # If ID is provided, assign it to a variable.
    # If no ID is provided, display an error in the browser.

    """if 'patient_id' in request.args:
        patient_id = request.args['patient_id']
    else:
        return "Error: No PatientId field provided. Please specify an id."
    """
    patient_id == request.view_args['patient_id']
    # Create an empty list for our results
    results = []
    # Loop through the data and match results that fit the requested ID.

    for id_ in range(len(df)):
        if df[id_]["patient_id"] == patient_id:
            results.append(df[id_])

    # Use the jsonify function from Flask to convert our list of
    # Python dictionaries to the JSON format.
    if len(results) < 1:
        return "PatientId is not found", 404
    else:
        return jsonify(results)


@app.route('/predict', methods=['GET'])
def all_unique_results():
    return jsonify(df_unique[:100])


@app.route('/predict/<patient_id>', methods=['GET'])
def unique_api_id(patient_id):
    # Check if an ID was provided as part of the URL.
    # If ID is provided, assign it to a variable.
    # If no ID is provided, display an error in the browser.
    """if 'patient_id' in request.args:
        patient_id = request.args['patient_id']
    else:
        return "Error: No PatientId field provided. Please specify an id."
    """
    patient_id == request.view_args['patient_id']
    # Create an empty list for our results
    results = []
    # Loop through the data and match results that fit the requested ID.

    for id_ in range(len(df_unique)):
        if df_unique[id_]["patient_id"] == patient_id:
            results.append(df_unique[id_])

    # Use the jsonify function from Flask to convert our list of
    # Python dictionaries to the JSON format.
    if len(results) < 1:
        return "PatientId is not found", 404
    else:
        return jsonify(results)


app.run(host=config["Service"]["Host"], port=int(config["Service"]["Port"]), debug=True)
