import pandas as pd
import matplotlib.pyplot as plt
import configparser
import preprocess, util
from config import config
import time
from sklearn.linear_model import RidgeClassifier

from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

import warnings
warnings.filterwarnings('ignore')

class ML_Model(object):
    def __int__(self, patient_df, prediction_path):#, prediction2_path):
        patient_df = self.patient_df
        prediction_path = self.prediction_path
        #prediction2_path = self.prediction2_path

    def model(self, patient_df, prediction_path):#, prediction2_path):
        patient_df = util.reduce_mem_usage(patient_df)

        # creating instance of labelencoder
        labelencoder = LabelEncoder()
        # Assigning numerical values and storing in another column
        patient_df.rename(columns={"id": "PatientId"}, errors="raise")
        patient_df['PatientId'] = labelencoder.fit_transform(patient_df['PatientId'])

        X_no = patient_df[patient_df.diagnosis != "opioid"]
        X_yes = patient_df[patient_df.diagnosis == "opioid"]

        X_yes_upsampled = X_yes.sample(n=len(X_no), replace=True, random_state=42)
        print(len(X_yes_upsampled))

        X_upsampled = X_no.append(X_yes_upsampled).reset_index(drop=True)

        X = X_upsampled.drop(['Opioid'], axis=1)  # features (independent variables)
        y = X_upsampled['Opioid']  # target (dependent variable)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # RidgeClassifier
        def ridge_classifier(X_train, y_train, X_test, y_test):
            clf_ridge = RidgeClassifier()  # create a ridge classifier object
            clf_ridge.fit(X_train, y_train)  # train the model

            ridge_importance = clf_ridge.coef_[0]
            # summarize feature importance
            for i, v in enumerate(ridge_importance):
                print('Feature: %0d, Score: %.5f' % (i, v))
            # plot feature importance
            plt.bar([x for x in range(len(ridge_importance))], ridge_importance)
            plt.show()

            ridge_pred_test = clf_ridge.predict(X_test)
            accuracy_score(y_test, ridge_pred_test)

            print(classification_report(y_test, ridge_pred_test))
            return ridge_pred_test

        # RandomForestClassifier
        def random_forest_classifier(X_train, y_train, X_test, y_test):

            n_estimators = config.get('RandomForestClassifier', 'n_estimators')
            max_depth = config.get('RandomForestClassifier', 'max_depth')
            min_samples_split = config.get('RandomForestClassifier', 'min_samples_split')
            min_samples_leaf = config.get('RandomForestClassifier', 'min_samples_leaf')

            hyper_random = {"n_estimators": n_estimators,
                            "max_depth": max_depth,
                            "min_samples_split": min_samples_split,
                            "min_samples_leaf": min_samples_leaf}

            start = time.time()
            print(start)
            clf_rf_tuned = GridSearchCV(RandomForestClassifier(), hyper_random,
                                        cv=5, verbose=1,
                                        n_jobs=-1)
            clf_rf_tuned.fit(X_train, y_train)

            end = time.time()
            hours, rem = divmod(end - start, 3600)
            minutes, seconds = divmod(rem, 60)
            print("\nProcess Time: " + "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))

            best_params_random = clf_rf_tuned.best_params_
            print(best_params_random)

            CV_clf_rf = RandomForestClassifier(max_depth=best_params_random["max_depth"],
                                               min_samples_leaf=best_params_random["min_samples_leaf"],
                                               min_samples_split=best_params_random["min_samples_split"],
                                               n_estimators=best_params_random["n_estimators"])

            CV_clf_rf.fit(X_train, y_train)
            y_test_predict_random = CV_clf_rf.predict_proba(X_test)[:, 1]
            yhat_random = CV_clf_rf.predict(X_test)
            fraction_of_positives, mean_predicted_value = calibration_curve(y_test, y_test_predict_random, n_bins=10)

            print(classification_report(y_test, yhat_random))
            return yhat_random, CV_clf_rf

        final_df = pd.DataFrame()
        final_df['PatientId'] = X_test.PatientId
        final_df['Actual_Diagnosis'] = y_test
        final_df['Ridge_Classifier_Probability'], CV_clf_rf = ridge_classifier(X_train, y_train, X_test, y_test)
        final_df['Random_Forest_Probability'] = random_forest_classifier(X_train, y_train, X_test, y_test)
        final_df.shape

        #CalibratedClassifierCV
        def calibrated_classifierCV(CV_clf_rf, X_train, y_train, X_test, y_test):
            start = time.time()

            # Create a corrected classifier.
            clf_sigmoid = CalibratedClassifierCV(CV_clf_rf, cv=10, method='sigmoid')
            clf_sigmoid.fit(X_train, y_train)
            y_test_predict_random_calibrated = clf_sigmoid.predict_proba(X_test)[:, 1]
            yhat_calibrated_random = clf_sigmoid.predict(X_test)
            fraction_of_positives, mean_predicted_value = calibration_curve(y_test, y_test_predict_random_calibrated, n_bins=10)

            end = time.time()
            hours, rem = divmod(end - start, 3600)
            minutes, seconds = divmod(rem, 60)
            print("\nProcess Time: " + "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))

            print(classification_report(y_test, yhat_calibrated_random))
            return yhat_calibrated_random, clf_sigmoid

        final_df['Calibrated_Random_Forest_Probability'], clf_sigmoid = calibrated_classifierCV(CV_clf_rf, X_train, y_train, X_test, y_test)
        print(classification_report(y_test, final_df['Ridge_Classifier_Probability']))
        print(classification_report(y_test, final_df['Random_Forest_Probability']))
        print(classification_report(y_test, final_df['Calibrated_Random_Forest_Probability']))

        # GaussianNaiveBias
        def gaussian_naive_bias(X_train, y_train, X_test, y_test):
            start = time.time()
            # Uncalibrated
            clf_nb = GaussianNB()
            clf_nb.fit(X_train, y_train)
            y_test_predict_nb = clf_nb.predict_proba(X_test)[:, 1]
            yhat_nb = clf_nb.predict(X_test)
            fraction_of_positives_nb, mean_predicted_value_nb = calibration_curve(y_test, y_test_predict_nb, n_bins=10)

            # Calibrated
            clf_sigmoid_nb = CalibratedClassifierCV(clf_nb, cv=10, method='isotonic')
            clf_sigmoid_nb.fit(X_train, y_train)
            y_test_predict_nb_calib = clf_sigmoid_nb.predict_proba(X_test)[:, 1]
            yhat_calibrated_nb = clf_sigmoid_nb.predict(X_test)
            fraction_of_positives_nb_calib, mean_predicted_value_nb_calib = calibration_curve(y_test, y_test_predict_nb_calib,
                                                                                              n_bins=10)

            # Calibrated, Platt
            clf_sigmoid_nb_calib_sig = CalibratedClassifierCV(clf_nb, cv=10, method='sigmoid')
            clf_sigmoid_nb_calib_sig.fit(X_train, y_train)

            y_test_predict_nb_calib_platt = clf_sigmoid_nb_calib_sig.predict_proba(X_test)[:, 1]
            yhat_calibrated_platt = clf_sigmoid_nb_calib_sig.predict(X_test)

            fraction_of_positives_nb_calib_platt, mean_predicted_value_nb_calib_platt = calibration_curve(y_test,
                                                                                                          y_test_predict_nb_calib_platt,
                                                                                                          n_bins=10)
            end = time.time()
            hours, rem = divmod(end - start, 3600)
            minutes, seconds = divmod(rem, 60)
            print("\nProcess Time: " + "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))
            return yhat_nb, yhat_calibrated_nb, yhat_calibrated_platt, clf_nb, clf_sigmoid_nb, clf_sigmoid_nb_calib_sig

        final_df['Naive_Bias_Probability'], final_df['Isotonic_Calibrated_Naive_Bias_Probability'], final_df['Sigmoid_Calibrated_Naive_Bias_Probability'], clf_nb, clf_sigmoid_nb, clf_sigmoid_nb_calib_sig = gaussian_naive_bias(X_train, y_train, X_test, y_test)


        print(classification_report(y_test, final_df['Naive_Bias_Probability']))
        print(classification_report(y_test, final_df['Isotonic_Calibrated_Naive_Bias_Probability']))
        print(classification_report(y_test, final_df['Sigmoid_Calibrated_Naive_Bias_Probability']))


        print(final_df.shape)

        # Train Results
        # Random Forest
        y_predict_train = CV_clf_rf.predict_proba(X_train)[:, 1]
        yhat_predict_train = CV_clf_rf.predict(X_train)

        # Calibrated Random Forest
        y_predict_crf_train = clf_sigmoid.predict_proba(X_train)[:, 1]
        yhat_predict_crf_train = clf_sigmoid.predict(X_train)

        # NB

        y_predict_nb_train = clf_nb.predict_proba(X_train)[:, 1]
        yhat_predict_nb_train = clf_nb.predict(X_train)

        # Isotonic
        y_predict_nb_isotonic_train = clf_sigmoid_nb.predict_proba(X_train)[:, 1]
        yhat_predict_isotonic_train = clf_sigmoid_nb.predict(X_train)

        # Sigmoid
        y_predict_nb_sigmoid_train = clf_sigmoid_nb_calib_sig.predict_proba(X_train)[:, 1]
        yhat_predict_sigmoid_train = clf_sigmoid_nb_calib_sig.predict(X_train)

        final_train_df = pd.DataFrame()
        final_train_df['PatientId'] = X_train.PatientId
        final_train_df['Actual_Diagnosis'] = y_train

        final_train_df['Random_Forest_Probability'] = y_predict_train
        final_train_df['Calibrated_Random_Forest_Probability'] = y_predict_crf_train
        final_train_df['Naive_Bias_Probability'] = y_predict_nb_train
        final_train_df['Isotonic_Calibrated_Naive_Bias_Probability'] = y_predict_nb_isotonic_train
        final_train_df['Sigmoid_Calibrated_Naive_Bias_Probability'] = y_predict_nb_sigmoid_train

        oud_df = final_df.append(final_train_df, ignore_index=True)

        oud_df['PatientId'] = labelencoder.inverse_transform(oud_df['PatientId'])
        oud_df2 = oud_df.groupby(['PatientId', 'Actual_Diagnosis'], as_index=False)['Random_Forest_Probability',
                                                                                    'Calibrated_Random_Forest_Probability',
                                                                                    'Naive_Bias_Probability',
                                                                                    'Isotonic_Calibrated_Naive_Bias_Probability',
                                                                                    'Sigmoid_Calibrated_Naive_Bias_Probability'].mean()
        oud_df2 = util.calculate_opioid_score(oud_df2)
        oud_df = util.calculate_opioid_score(oud_df)

        cols = oud_df.columns.tolist()
        cols = cols[:2] + cols[-1:] + cols[2:-1]

        oud_df = oud_df[cols]
        oud_df2 = oud_df2[cols]
        oud_df['Prediction_Source'] = "ml_model"
        oud_df2['Prediction_Source'] = "ml_model"

        patient_df = oud_df[['PatientId', 'Opioid_Score', 'Opioid_Risk', 'Prediction_Source']]
        patient_df2 = oud_df2[['PatientId', 'Opioid_Score', 'Opioid_Risk', 'Prediction_Source']]

        #return patient_df, patient_df2
        patient_df.to_json(prediction_path)
        #patient_df2.to_json(prediction2_path)

def risk_calculator(df, flag):
    print('++++++++++++++++++')
    print(df[0])
    print('++++++++++++++++++')
    temp_pred_df = pd.DataFrame(columns = ['Patient_Id', 'Opioid_Score', 'Opioid_Risk', 'Prediction_Source'])

    calculated_score = util.calc_oud_score(patient_df)
    scaled_risk = util.risk_calculator(calculated_score)
    patient_id = df[0]
    temp_pred_df['Patient_Id'] = patient_id
    temp_pred_df['Opioid_Score'] = calculated_score
    temp_pred_df['Opioid_Risk'] = scaled_risk
    temp_pred_df['Prediction_Source'] = flag

    return temp_pred_df


if __name__ == "__main__":
    start = time.time()

    config = configparser.ConfigParser()
    config.read_file(open(r'config.ini'))
    path1 = config.get('Case_Path', 'case1_path')
    path2 = config.get('Case_Path', 'case2_path')
    path3 = config.get('Case_Path', 'case3_path')
    path4 = config.get('Case_Path', 'case4_path')
    prediction_path = config.get('Case_Path', 'prediction_path')
    #prediction2_path = config.get('Case_Path', 'prediction2_path')

    path_list = [path1, path2, path3, path4]
    flag = config.get('Model_Type', 'flag')

    prediction_df = pd.DataFrame(columns=["Patient_Id", "Opioid_Score", "Opioid_Risk", "Prediction_Source"])
    #patient_df = util.read_postgre_data()
    #patient_df = patient_df.rename(columns={"patient_id": "Patient_Id"})

    if flag == 'ml':
        for path in path_list:
            patient_df = preprocess.data_preprocess(path)
            ML_Model(patient_df, prediction_path)#, prediction2_path)
    else:
        patient_df = util.read_postgre_data()
        temp_pred_df = pd.DataFrame(columns = ['Patient_Id', 'Opioid_Score', 'Opioid_Risk', 'Prediction_Source', 'Created_On'])

        records_to_insert = []
        for index, row in patient_df.iterrows():
            patient_age = util.age(row.dateOfBirth)
            row['age'] = patient_age

            calculated_score = util.calc_oud_score(row)
            scaled_risk = util.risk_calculator(calculated_score)

            row['Opioid_Score'] = calculated_score
            row['Opioid_Risk'] = scaled_risk
            row['Prediction_Source'] = flag
            temp_pred_df = temp_pred_df.append(row[['patient_id', 'Opioid_Score', 'Opioid_Risk', 'Prediction_Source', 'Created_On']])
        util.write_data_postgre(temp_pred_df)

    end = time.time()
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("\nÇalışma süresi: " + "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))