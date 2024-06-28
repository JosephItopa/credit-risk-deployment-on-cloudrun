import datetime
import numpy as np
import pandas as pd

region_dict = {'london': 0, 'west midlands': 1, 'south east': 2,\
                'east anglia': 3, 'yorkshire': 4, 'east midlands': 5,\
                'north east': 6, 'south west': 7, 'north west': 8,\
                'wales': 9, 'northern ireland': 10, 'scotland':11}


industry_dict = {'other services': 0, 'admin & support': 1,\
                 'construction': 2, 'retail trade, except of motor vehicles': 3,\
                 'multiple industries': 4}
approved_term_dict = {
                      '24 Months': 0, '36 Months': 1, '12 Months': 2, '18 Months': 3, '3 Months': 4, '15 Months': 5,\
                      '9 Months': 6, '4 Months': 7, '6 Months': 8, '48 Months': 9, '1 Months': 10, '20 Months': 11,\
                      '10 Months': 12, '30 Months': 13, '16 Months': 14, '21 Months': 15, '14 Months': 16, '19 Months': 17,\
                      '60 Months': 18, '8 Months': 19, '13 Months': 20, '5 Months': 21, '2 Months': 22, '38 Months': 23,\
                      '7 Months': 24, '26 Months': 25, '35 Months': 26, '0 Months': 27, '42 Months': 28, '28 Months': 29
                    }
# loan_status_dict.get(loan_status)

model_data_clf = {}

model_data_rgr = {}

other_services = ["utilities", "financial & insurance", "wholesale & retail & repair of motor vehicles", "arts, entertainment & recreation"
                  "transportation", "human health", "unknown", "manufacturing", "information & communication", "professional, scientific & technical", "manufacturing"]

def replace_industry(packet)->dict:
    industry = packet["industry"].lower()
    if industry in other_services:
        packet["industry"] = "other services"
    elif industry == "public services":
        packet["industry"] = "admin & support"
    elif industry == "real estate":
        packet["industry"] = "construction"
    elif industry == "accommodation & food":
        packet["industry"] = "arts, entertainment & recreation"
    elif industry == "wholesale trade, except of motor vehicles":
        packet["industry"] = "retail trade, except of motor vehicles"
    elif industry == "agriculture & mining":
        packet["industry"] = "multiple industries"
    else:
        packet["industry"] = industry
    return packet

def preprocessor_clf(data):
    data = replace_industry(data)
    try:
        model_data_clf['applied_loan_type_id'] = int(data['applied_loan_type_id'])
    except Exception as e:
        return "applied_loan_type_id should be integer"
    
    try:
        model_data_clf['loan_amount'] = round(float(data['loan_amount'])/1000,2)
    except Exception as e:
        return "loan_amount should be float"
    
    try:
        model_data_clf['loan_funding_purpose_id'] = int(data['loan_funding_purpose_id'])
    except Exception as e:
        return "loan_funding_purpose_id should be integer"

    try:
        model_data_clf['approved_term'] = approved_term_dict.get(str(data['approved_term']))
    except Exception as e:
        return "approved_term should be a string"
    
    try:
        model_data_clf['industry'] = industry_dict.get(str(data['industry']))
    except Exception as e:
        return "industry should be a string"

    try:
        model_data_clf['best_rn'] = round(float(data['best_rn']), 2)
    except Exception as e:
        return "best_rn should be float"
    
    try:
        model_data_clf['active_directors'] = float(data['active_directors'])
    except Exception as e:
        return "active_directors should be integer"

    try:
        model_data_clf['active_personal_guarantors_(loan)'] = int(data['active_personal_guarantors_(loan)'])
    except Exception as e:
        return "active_personal_guarantors should be a integer"

    try:
        model_data_clf['company_searches_(4-12m)'] = int(data['company_searches_(4-12m)'])
    except Exception as e:
        return "company_searches_(4-12m) should be integer"

    try:
        model_data_clf['g1_call_credit_score'] = int(data['g1_call_credit_score'])
    except Exception as e:
        return "g1_call_credit_score should be integer"

    try:
        model_data_clf['g1_searches_(0-1m)'] = int(data['g1_searches_(0-1m)'])
    except Exception as e:
        return "g1_searches_(0-1m) should be integer"

    try:
        model_data_clf['g1_searches_(2-3m)'] = int(data['g1_searches_(2-3m)'])
    except Exception as e:
        return "g1_searches_(2-3m) should be integer"

    try:
        model_data_clf['no_of_directors'] = int(data['no_of_directors'])
    except Exception as e:
        return "no_of_directors should be integer"

    try:
        model_data_clf['no_of_shareholders'] = int(data['no_of_shareholders'])
    except Exception as e:
        return "no_of_shareholders should be integer"

    model_features = ['applied_loan_type_id','loan_amount','loan_funding_purpose_id','approved_term',\
                      'industry','best_rn','active_directors','active_personal_guarantors_(loan)','company_searches_(4-12m)',\
                      'g1_call_credit_score','g1_searches_(0-1m)','g1_searches_(2-3m)','no_of_directors','no_of_shareholders']
    
    model_data_df = pd.DataFrame([model_data_clf], columns = model_features)
    model_data_df.fillna(0, inplace = True)
    model_data_df = list(model_data_df.iloc[0])
    print(model_data_df)
    return model_data_df

def preprocessor_rgr(data):
    data = replace_industry(data)
    # print(data)
    try:
        model_data_rgr['applied_loan_type_id'] = int(data['applied_loan_type_id'])
    except Exception as e:
        return "applied_loan_type_id should be integer"
    
    try:
        model_data_rgr['loan_amount'] = round(float(data['loan_amount'])/1000,2)
    except Exception as e:
        return "loan_amount should be float"
    
    try:
        model_data_rgr['loan_funding_purpose_id'] = int(data['loan_funding_purpose_id'])
    except Exception as e:
        return "loan_funding_purpose_id should be integer"
    
    try:
        model_data_rgr['approved_term'] = approved_term_dict.get(str(data['approved_term']))
    except Exception as e:
        return "approved_term should be a string"

    try:
        model_data_rgr['employees'] = int(data['no_of_employees'])
    except Exception as e:
        return "employees should be float"
    
    try:
        model_data_rgr['region'] = region_dict.get(str(data['region']))
    except Exception as e:
        return "region should be a string"

    try:
        model_data_rgr['active_directors'] = int(data['active_directors'])
    except Exception as e:
        return "active_directors should be integer"
    
    try:
        model_data_rgr['best_rn'] = round(float(data['best_rn']), 2)
    except Exception as e:
        return "best_rn should be float"

    try:
        model_data_rgr['active_personal_guarantors_(loan)'] = int(data['active_personal_guarantors_(loan)'])
    except Exception as e:
        return "active_personal_guarantors should be a integer"

    try:
        model_data_rgr['company_searches_(0-1m)'] = int(data['company_searches_(0-1m)'])
    except Exception as e:
        return "company_searches_(4-12m) should be integer"

    try:
        model_data_rgr['company_searches_(2-3m)'] = int(data['company_searches_(2-3m)'])
    except Exception as e:
        return "company_searches_(2-3m) should be integer"

    try:
        model_data_rgr['company_searches_(4-12m)'] = int(data['company_searches_(4-12m)'])
    except Exception as e:
        return "company_searches_(4-12m) should be integer"
    
    try:
        model_data_rgr['g1_call_credit_score'] = round(float(data['g1_call_credit_score']), 2)
    except Exception as e:
        return "g1_call_credit_score should be float"
    
    try:
        model_data_rgr['g1_active_ccjs'] = float(data['g1_active_ccjs'])
    except Exception as e:
        return "g1_equifax_valuation should be float"
    
    try:
        model_data_rgr['g1_searches_(2-3m)'] = int(data['g1_searches_(2-3m)'])
    except Exception as e:
        return "g1_searches_(2-3m) should be integer"
    
    try:
        model_data_rgr['business_type_id'] = int(data['business_type_id'])
    except Exception as e:
        return "business_type_id should be integer"

    try:
        model_data_rgr['no_of_directors'] = int(data['no_of_directors'])
    except Exception as e:
        return "no_of_directors should be integer"

    try:
        model_data_rgr['no_of_shareholders'] = int(data['no_of_shareholders'])
    except Exception as e:
        return "no_of_shareholders should be integer"
    
    try:
        model_data_rgr['monthofyear'] = int(data['monthofyear']) #int(date.isocalendar().week)
    except Exception as e:
        return e

    model_features = ['applied_loan_type_id','loan_amount','loan_funding_purpose_id',\
                    'approved_term','employees','region','active_directors','active_personal_guarantors_(loan)','company_searches_(0-1m)',\
                    'company_searches_(2-3m)', 'company_searches_(4-12m)','g1_call_credit_score','g1_active_ccjs','g1_searches_(2-3m)','business_type_id',\
                    'no_of_directors','no_of_shareholders','monthofyear']

    model_data_df = pd.DataFrame([model_data_rgr], columns = model_features)
    model_data_df.fillna(0, inplace = True)
    model_data_df = list(model_data_df.iloc[0])
    print(model_data_df)
    return model_data_df