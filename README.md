# Opioid Usage Addiction Probability

The task is to predict the probability of opioid usage addiction for each patient.

This project was prepared with two steps. One of them is model creation, the other one is servis api. The output of machine learning algorithms were written in both sqlite as a database and csv as a file.
SQLite was chosen as the database because this project did not have a very complex data set and focused on that was paid to the ease of installation. Compared to other databases, SQLite performs lower performance, but high performance is not expected from a dataset in this project.

After prediction, you can create rest api to see results. "get_service_opioid.py" folder was created for rest service. In this step for easy and fast execution, I prefered to dockerize the service. For dockerization, you have to run below commands on terminal.

1) docker build --tag cliexa-ai-service-app:1.0 .
2) docker run -p 8080:8080 --name cliexa-ai-service-app cliexa-ai-service-app:1.0

After this process, you can use postman to test. There are four different get service under two main group. 

**Patient Service:**

url:8080/patients_list

def:This service return the list of all patients id

**Group one:**

a-(predict_all_patients_last_statu) :

url:8080/results

Def:This service return probability value for every transaction. This method doesn't need any parameter. 

b-(predict_unique_patient_status) :

url:8080/results/<patient_id>

def:This service get userId as an input parameter and return transactional results of specific user


**Group two:**

a-(predict_all_patients_last_statu) :

url:8080/predict

def:This service return patient based probability value. This method doesn't need any parameter. 

b-(predict_unique_patient_last_statu) :

url:8080/predict/<patient_id>

def:This service get PatientId as an input parameter and return his/her result of oud addiction probability

Whole services return dataframe as a json message.

You can find postman collection on collection folder.
