from operator import not_
from flask import Flask, render_template, request
import xgboost as xgb
import numpy as np
import pandas as pd
import sklearn

app = Flask(__name__)

classifier = xgb.Booster()
classifier.load_model("/Users/yashgupta/Desktop/BerTrugS-main/backends_final/updated_final_model_5.bin")




def funcfunc(step, type, amount,oldbalanceOrg,newbalanceOrig,oldbalanceDest,newbalanceDest):    
    if type=="TRANSFER" or type=="CASH_OUT":
        if type=="TRANSFER" :
            type=0
        else:
            type=1
        if oldbalanceDest==0 and newbalanceDest == 0 and amount!=0:
            oldbalanceDest=-1
            newbalanceDest=-1
        if oldbalanceOrg==0 and newbalanceOrig == 0 and amount!=0:
            oldbalanceOrg=np.nan
            newbalanceOrig=np.nan
        errorbalanceDest=oldbalanceDest+amount-newbalanceDest
        errorbalanceOrig=newbalanceOrig+amount-oldbalanceOrg

        return [step,type,amount,oldbalanceOrg,newbalanceOrig,oldbalanceDest,newbalanceDest,errorbalanceDest,errorbalanceOrig]
    else:
        return -1
def get_list(idinp):
    df = pd.read_csv("./lastten.csv")
    filtered_list = df[df['nameOrig'] == idinp].values.tolist()
    if not filtered_list:  # Check if the list is empty
        return None  # Or return an empty list [], or some default value, or raise a custom error
    return filtered_list[0]


def predict(idinp):
    listlist =get_list(idinp)
    if listlist is None:  # Check if listlist is None
        return -1, "No data available"  # Adjust based on how you want to handle this case

    listlist2=funcfunc(listlist[0],listlist[1],listlist[2],listlist[4],listlist[5],listlist[7],listlist[8])
    
    if(listlist2==-1):
        return 0,100

    onerow = pd.DataFrame({'step': [listlist2[0]], 'type': [listlist2[1]], 'amount': [listlist2[2]], 'oldbalanceOrg': [listlist2[3]]
                       , 'newbalanceOrig': [listlist2[4]], 'oldbalanceDest': [listlist2[5]], 'newbalanceDest': [listlist2[6]], 'errorbalanceDest': [listlist2[7]]
                       , 'errorbalanceOrig': [listlist2[8]]})
    
    prediction_prob =  classifier.inplace_predict(onerow)


    if prediction_prob < 0.5:
        ans=0
        not_fraud_probability=1-prediction_prob
        return ans,not_fraud_probability
    else:
        ans=1
        fraud_probability=prediction_prob
        return ans,fraud_probability
    

# @app.route('/',methods=['POST'])
# def Home():
#     data = request.json
#     response=dict()
#     other_details=get_list(data['id'])
#     response['other_details']=other_details
  
#     prediction=predict(data['id'])
#     try:
#         response['prediction']=str(prediction[0])
#         response['probability']=str(prediction[1][0])
#     except:
#         print(prediction)
#         response['prediction']=str(prediction[0])
#         response['probability']=str(prediction[1])
#     print(response)
#     return response

# if __name__=="__main__":
#     app.run(debug=True)
    
# error handling 
# app = Flask(__name__)

# classifier = xgb.Booster()
# classifier.load_model("/Users/yashgupta/Desktop/BerTrugS-main/backends_final/updated_final_model_5.bin")

# Your existing function definitions like funcfunc, get_list, predict...

@app.route('/', methods=['POST'])
def Home():
    try:
        data = request.json
        other_details = get_list(data['id'])

        # Immediately check if other_details is None and return an error before proceeding
        if other_details is None:
            return {"error": "No details found for the provided ID"}, 404

        # Assuming other_details is not None, continue with your logic
        prediction_result = predict(data['id'])  # Make sure predict can handle empty or None inputs safely
        response = {
            'other_details': other_details,
            'prediction': str(prediction_result[0]),
            'probability': str(prediction_result[1])
        }
        return response
    except Exception as e:
        return {"error": str(e)}, 500

if __name__ == "__main__":
    app.run(debug=True)