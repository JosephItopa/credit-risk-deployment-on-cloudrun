# import modules
import asyncio
from fastapi import FastAPI, Request
from predict import deposit_amount, loan_default

# instantiate fastapi
app= FastAPI()

loan_status_dict = {1: "default", 0:"no_default"}

def check_decile(value):
    """generate decile value and decile probability of default"""
    if value <= 0.16:
        decile, proba_ = 0, 0.083
    elif value > 0.16 and value <= 0.26:
        decile, proba_ = 1, 0.196
    elif value > 0.26 and value <= 0.38:
        decile, proba_ = 2, 0.31
    elif value > 0.38 and value <= 0.54:
        decile, proba_ = 3, 0.445
    elif value > 0.54 and value <= 0.74:
        decile, proba_ = 4, 0.655
    elif value > 0.74 and value <= 0.85:
        decile, proba_ = 5, 0.835
    elif value > 0.85 and value <= 0.90:
        decile, proba_ = 6, 0.923
    elif value > 0.90 and value <= 0.95:
        decile, proba_ = 7, 0.9709
    elif value > 0.95 and value <= 0.98:
        decile, proba_ = 8, 0.989
    elif value > 0.98:
        decile, proba_ = 9, 1.000000
    return decile, proba_

def model_predict(df):
    """produce predictions for loan default status, deposited amount, and estimations"""
    total_min_margin = 0
    min_required_profit = 0
    finance_cost_sonia = 0
    finance_cost_margin = 0
    contribution_cost = 0

    minRequiredProfit = 0.053
    financeCostSonia = 0.049
    financeCostMargin = 0.05
    contributionCost = 0.053

    if df['minRequiredProfit'] == 0.0:
        min_required_profit = minRequiredProfit
    else:
        min_required_profit = df['minRequiredProfit']

    if df['financeCostSonia'] == 0.0:
        finance_cost_sonia = financeCostSonia
    else:
        finance_cost_sonia = df['financeCostSonia']

    if df['financeCostMargin'] == 0.0:
        finance_cost_margin = financeCostMargin
    else:
        finance_cost_margin = df['financeCostMargin']

    if df['contributionCost'] == 0.0:
        contribution_cost = contributionCost
    else:
        contribution_cost = df['contributionCost']

    
    # total_min_margin 
    adjusted_interest_rate = min_required_profit + finance_cost_sonia + finance_cost_margin + contribution_cost

    clf_data, loan_status_, loan_proba = loan_default(df)
    print(loan_proba)
    loan_status = [1 if loan_proba > 0.5 else 0][0]

    decile_, bucket_loan_proba = check_decile(loan_proba)

    rgr_data, predicted_amount = deposit_amount(df)

    predicted_amount = round(predicted_amount*1000,2) # clf_data, rgr_data

    loss_given_default = 1 # for a loan term of one year, assuming clients make the payment for only the first two months

    total_loan = round(predicted_amount + (adjusted_interest_rate * predicted_amount), 2)

    expected_loss_default = total_loan * bucket_loan_proba * loss_given_default #(loan_proba/region_avg_app.get(df["region"]))

    clf_columns = ['applied_loan_type_id','loan_amount','loan_funding_purpose_id','approved_term',
                      'industry','best_rn','active_directors','active_personal_guarantors_(loan)','company_searches_(4-12m)',
                      'g1_call_credit_score','g1_searches_(0-1m)','g1_searches_(2-3m)','no_of_directors','no_of_shareholders']
    clf_inpt = dict(zip(clf_columns, clf_data[0]))
    rgr_columns = ['applied_loan_type_id','loan_amount','loan_funding_purpose_id',
                    'approved_term','employees','region','active_directors','active_personal_guarantors_(loan)','company_searches_(0-1m)',
                    'company_searches_(2-3m)', 'company_searches_(4-12m)','g1_call_credit_score','g1_active_ccjs','g1_searches_(2-3m)','business_type_id',
                    'no_of_directors','no_of_shareholders','monthofyear']
    rgr_inpt = dict(zip(rgr_columns, rgr_data[0]))

    pred_result = {"total_loan_amount": round(total_loan, 2), "predict_loan_amount": round(predicted_amount, 2),\
                   "adjusted_interest_amount": round(adjusted_interest_rate*predicted_amount, 2), "adjusted_interest_rate": round(adjusted_interest_rate*100, 2),\
                   "exposure_at_default":round(expected_loss_default, 2),\
                   "probability_of_default": round(loan_proba*100, 0), "bucket_probability_of_default": round(bucket_loan_proba*100,0),\
                   "bucket_decile": decile_, "probability_status": loan_status_dict.get(loan_status),\
                   "contribution_cost": round(predicted_amount*contribution_cost, 2), "finance_cost_margin": round(predicted_amount*finance_cost_margin, 2),\
                   "finance_cost_sonia": round(predicted_amount*finance_cost_sonia, 2), "min_required_profit": round(predicted_amount*min_required_profit, 2),\
                   "processed_input_clf":clf_inpt, "processed_input_rgr":rgr_inpt}

    return pred_result

@app.get("/")
async def get_input(request:Request):
    """
        Get inputs from users and call the simple interest function to evaluate
        the parameters then return the output. 
    """
    getInput = await request.json()

    loop = asyncio.get_event_loop()

    predictor = model_predict(getInput) #

    return predictor