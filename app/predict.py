import pickle
import pandas as pd
import numpy as np
from .preprocessor import preprocessor_clf, preprocessor_rgr

loan_status = ""
loan_amount = 0
status_proba = 0

def deposit_amount(df):

    input_data = preprocessor_rgr(df)
    
    data = np.array(input_data).reshape(1, 18)
    file_name = './models/deposit_amount_gbr_v1.1.1.pkl'
    rgr = pickle.load(open(file_name , 'rb'))

    return data, round(rgr.predict(data)[0], 2)

def loan_default(df):

    input_data = preprocessor_clf(df)

    data = np.array(input_data).reshape(1, 14)
    file_name = './models/loan_default_rfr_v1.1.1.pkl'
    clf = pickle.load(open(file_name , 'rb'))

    loan_status, status_proba = clf.predict(data)[0], clf.predict_proba(data)[:, 1][0]
    return data, loan_status, round(status_proba, 3)