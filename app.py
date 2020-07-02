import pickle
from flask import Flask, request, render_template, url_for
import pandas as pd
import numpy as np

app = Flask(__name__)
clf = pickle.load(open("model.pkl", "rb"))

@app.route('/')
def home():
    #getting template for input page
    
    return render_template('home.html') 

@app.route('/predict', methods = ['GET'])
def prediction():
    #getting values from input screen
    
    de = request.args.get("de")
    dr = request.args.get("dr")
    qd = int(request.args.get("qd"))
    qr = int(request.args.get("qr"))
    q_return = int(request.args.get("q_return"))
    
    #check for return quantity > received quantity
    
    if q_return > qr:
        return ("Invalid input: Quantity returned cannot be greater than quantity received")
    
    #scaling for promptness
    
    daydiff = (pd.to_datetime(de) - pd.to_datetime(dr)).days
    if daydiff > 0:
        daydiff = daydiff/10
    file = pd.read_csv('compiled.csv',usecols=['DeliveryTime','EnteredReceivedQuantity','QuantityDemandedFinal'])
    file = file.append({'DeliveryTime': daydiff}, ignore_index = True)
    file['DeliveryTime'] = file['DeliveryTime'].apply(abs)
    file['DeliveryTime'] = -file['DeliveryTime']
    promptness = (daydiff - np.min(file['DeliveryTime']))/(np.max(file['DeliveryTime']) - np.min(file['DeliveryTime']))
    
    quantity = qr/qd
    
    #scaling for quality and check for quantity received is 0
    
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
    
    final_score = round((0.5*quality + 0.3*promptness + 0.2*quantity)*5,2)  # Rating of the vendor out 5; weights 0.5,0.3,0.2 for quality, qty, promptness
    
    #settings the labels correctly
    label_dict = {}
    test = [[[1,1,1]],[[0,1,1]],[[1,0,1]],[[1,1,0]]]
    label_dict['Performing'] = clf.predict(test[0])[0]
    label_dict['quantity'] = clf.predict(test[1])[0]
    label_dict['promptness'] = clf.predict(test[2])[0]
    label_dict['quality'] = clf.predict(test[3])[0]
    
    #probablitlites for quntity, promptness and quality of given input

    result = clf.predict_proba([[quantity, promptness, quality]])[0]
    
    new = [['Issue with Quantity is ', result[label_dict['quantity']]], ['Issue with Promptness is ', result[label_dict['promptness']]], ['Issue with Quality is ', result[label_dict['quality']]]]
    if clf.predict([[quantity, promptness, quality]]) == label_dict['Performing']:
        answer = "Performing"
        new.sort(key = lambda x: x[1])
    else:
        answer = "Non-performing"
        new.sort(key = lambda x: x[1], reverse = True)
    for i in range(0,3):
        new[i][1]=round(new[i][1]*100,2)
    return render_template("result.html", answer = answer, value = new, rating = str(final_score)+' out of 5', value2='%')
    
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
