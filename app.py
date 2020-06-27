import pickle
from flask import Flask, request, render_template, url_for
import pandas as pd
import numpy as np
#import flasgger
from flasgger import Swagger

app = Flask(__name__)
Swagger(app)
model = pickle.load(open("model.pkl", "rb"))

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
        quality = 0
    #print(promptness, quantity, quality)
    prediction = model.predict([[quantity, promptness, quality]])
    my_dict = {0: 'Performing Vendor', 1: 'Non-performing Vendor: Quantity Issue', 2: 'Non-performing Vendor: Quality Issue', 3: 'Non-performing Vendor: Promptness Issue'}
    #return("Anwer is " + str(*prediction))
    return render_template("result.html", value = my_dict[int(*prediction)])

if __name__ == "__main__":
    app.run()
    #app.run(debug=True,host='0.0.0.0',port=5000)