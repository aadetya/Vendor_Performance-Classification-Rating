import pandas as pd
import numpy as np
#from numpy import linalg as LA
from sklearn.preprocessing import MinMaxScaler
#from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
#import datetime as dt
import pickle
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('compiled.csv')

#df.head()

#df.info()
# RemainingQty = EnteredReceivedQuantity - QuantityDemandedFinal
# DeliveryTime = EarlyDeliveryDate - ReceivedDate

# Because there are some null values in PercentOfQuantityReturned
df.fillna(0,inplace=True)
#df.info()

df['EarlyDeliveryDate'] = pd.to_datetime(df['EarlyDeliveryDate'])
df['ReceivedDate'] = pd.to_datetime(df['ReceivedDate'])
#df.info()
    
df['DeliveryTime'] = [x/10 if x>0 else x for x in df.DeliveryTime]

df['NormDeliveryTime'] = 0-df.DeliveryTime.abs()
scale = MinMaxScaler()
df['NormDeliveryTime'] = scale.fit_transform(df[['NormDeliveryTime']])

df['PercentKept'] = 1-df.PercentOfQuantityReturned


df['ReceivedMonth'] = df['ReceivedDate'].dt.month
df['ReceivedYear'] = df['ReceivedDate'].dt.year
grouped_monthly = df.groupby(['ReceivedYear', 'ReceivedMonth', 'Vendor']).agg(MonthlyNormDeliveryTime = pd.NamedAgg(column = 'NormDeliveryTime', aggfunc='mean'), MonthlyPercentReceived = pd.NamedAgg(column = 'PercentOfQuantityReceived', aggfunc='mean'), MonthlyPercentKept = pd.NamedAgg(column = 'PercentKept', aggfunc='mean'), VendorId = pd.NamedAgg(column = 'Vendor', aggfunc='first'))



"""*Categorization* of orders into performing and non-performing along with the reason. (Promptness, Quantity, Quality)"""


cls = KMeans(n_clusters = 4)
cls_assignment = cls.fit_predict(grouped_monthly[['MonthlyPercentReceived','MonthlyNormDeliveryTime','MonthlyPercentKept']])
grouped_monthly['label'] = cls_assignment


grouped = grouped_monthly.groupby("VendorId")
vendor_output = pd.DataFrame(columns=['Vendor_ID', 'Performance', 'Performance_Percent','UnderPerformance (Quality)', 'UnderPerformance (Quantity)', 'UnderPerformance (Promptness)'])
for name, group in grouped:
    values = group['label'].value_counts()
    values1 = values
    for i in range(0,4):
        if i not in values1.index:
            values1[i] = 0
    total = np.sum(group['label'].value_counts())

    for i,value in enumerate(values):
        values1[values[values == value].index] = value/total

    maxperc_index = values1[values1 == np.max(values1)].index[0]
    mydict = {0:'Performing', 1: 'Quality Issue', 2: 'Quantity Issue', 3: 'Promptness Issue'}
    
    if maxperc_index == 0 and values1[maxperc_index] < 0.75:
            maxperc_index = values1.sort()[-2].index[0]
    
    vendor_output = vendor_output.append({'Vendor_ID': group['VendorId'][group['VendorId'].first_valid_index()],'Performance': mydict[maxperc_index], 'Performance_Percent': values1[0],'UnderPerformance (Quantity)': values1[2],'UnderPerformance (Promptness)' : values1[3], 'UnderPerformance (Quality)': values1[1]}, ignore_index=True)
#vendor_output


clf = RandomForestClassifier()
X = grouped_monthly[['MonthlyPercentReceived','MonthlyNormDeliveryTime','MonthlyPercentKept']]
y = grouped_monthly['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
clf.fit(X_train,y_train)

#clf.fit(X,y)
pickle.dump(clf, open('model.pkl', 'wb'))

#print(pd.Series(clf.predict(X)).value_counts())
#print(grouped_monthly.label.value_counts())
clf.score(X_train,y_train),clf.score(X_test,y_test)
pred = clf.predict(X_test)
print(clf.predict([[1,0,1]]))