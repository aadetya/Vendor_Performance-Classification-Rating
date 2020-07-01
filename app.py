import pickle
from flask import Flask, request, render_template, url_for
import pandas as pd
import numpy as np
#import flasgger
from flasgger import Swagger

app = Flask(__name__)
Swagger(app)
clf = pickle.load(open("model.pkl", "rb"))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods = ['GET'])
def prediction():
    de = request.args.get("de")
    dr = request.args.get("dr")
    qd = int(request.args.get("qd"))
    qr = int(request.args.get("qr"))
    q_return = int(request.args.get("q_return"))
    
    if q_return > qr:
        return ("Invalid input: Quantity returned cannot be greater than quantity received")
    
    daydiff = (pd.to_datetime(de) - pd.to_datetime(dr)).days
    if daydiff > 0:
        daydiff = daydiff/10
    print(daydiff)
    file = pd.read_csv("compiled.csv", usecols =['DeliveryTime'])
    file = file.append({'DeliveryTime': daydiff}, ignore_index = True)
    file['DeliveryTime'] = file['DeliveryTime'].apply(abs)
    file['DeliveryTime'] = -file['DeliveryTime']
    promptness = (daydiff - np.min(file['DeliveryTime']))/(np.max(file['DeliveryTime']) - np.min(file['DeliveryTime']))
    quantity = qr/qd
    try:
        quality = 1 - q_return/qr
    except ZeroDivisionError:
        quality = 1
    
    ## OLD VERSION
        #print(promptness, quantity, quality)
        #prediction = model.predict([[quantity, promptness, quality]])
        #my_dict = {0: 'Performing Vendor', 1: 'Non-performing Vendor: Quantity Issue', 2: 'Non-performing Vendor: Quality Issue', 3: 'Non-performing Vendor: Promptness Issue'}
        #return("Anwer is " + str(*prediction))
    
    
    ## NEW VERSION
    
    final_score = (0.5*quality + 0.3*promptness + 0.2*quantity)*5   # Rating of the vendor out 5; weights 0.5,0.3,0.2 for quality, qty, promptness
    
    #settings the labels correctly
    label_dict = {}
    test = [[[1,1,1]],[[0,1,1]],[[1,0,1]],[[1,1,0]]]
    label_dict[clf.predict(test[0])[0]] = 'Performing'
    label_dict[clf.predict(test[1])[0]] = 'Non-Performing: Issue with Quantity'
    label_dict[clf.predict(test[2])[0]] = 'Non-Performing: Issue with Promptness'
    label_dict[clf.predict(test[3])[0]] = 'Non-Performing: Issue with Quality'

    result = clf.predict_proba([[quantity, promptness, quality]])[0]
    
    new = [('quantity', result[1]), ('promptness', result[2]), ('quality', result[3])]
    if final_score > 3:
        answer = "Performing"
        new.sort(key = lambda x: x[1])
    else:
        answer = "Non-performing"
        new.sort(key = lambda x: x[1], reverse = True)
    
    return render_template("result.html", answer = answer, value = new, rating = final_score)
    
# THRESHOLD = 0.25
#     l = [[quantity, promptness, quality]]
#     result_list = []
#     result = clf.predict_proba(l)[0]
#     factor_ratings = [quantity, promptness, quality]

#     for i in range(len(result)):
#         if result[i]>THRESHOLD:
#             result_list.append(label_dict[i])
#     if 'Performing' in result_list and len(result_list)>1:
#         result_list.remove('Performing')
#     return render_template("result.html", value = result_list)

if __name__ == "__main__":
    app.run()
    #app.run(debug=True,host='0.0.0.0',port=5000)
