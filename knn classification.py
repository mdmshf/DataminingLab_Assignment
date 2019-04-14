import re
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
model= RandomForestClassifier(n_estimators=10)
import numpy as np
df_list = []
print("Starting training of the data")
with open('training_data') as f:
    for line in f:
        
        line = line.strip()
        
        columns = re.split(':', line, maxsplit=4)
        df_list.append(columns)
df = pd.DataFrame(df_list)
#print df[0]
#print df[1]
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(df[1])
#print X_train_counts
#print X_train_counts.shape
tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
X_train_tf = tf_transformer.transform(X_train_counts)
#print X_train_tf.shape
print("Training completed\n")
#df[0].target[:10]
y=[]
for i in range(len(df)):
    temp=df[0][i]
    y.append(temp[-1])
Y=np.array(y)
print("Generating the KNN model\n")
#model
from sklearn.neighbors import KNeighborsRegressor
neigh = KNeighborsRegressor(n_neighbors=2) 
model.fit(X_train_tf,Y)
##fitting is done

print("Started testing the data")
df_list = []
with open('testing_data') as f:
    for line in f:
        line = line.strip()
        columns = re.split(':', line, maxsplit=4)
        df_list.append(columns)
docs_new = pd.DataFrame(df_list)
y_test=[]
for i in range(len(docs_new)):
    temp=df[0][i]
    y_test.append(temp[-1])
Y_test=np.array(y_test)


X_test_counts = count_vect.transform(docs_new[1])
tf_transformer = TfidfTransformer(use_idf=False)
X_new_tfidf = tf_transformer.transform(X_test_counts)
print("Calculating Accuracy of the model\n")
predicted= model.predict(X_new_tfidf)
#print predicted
print("Prediction completed\n")
from sklearn import metrics
# testing score
crosstab=pd.crosstab(predicted,Y_test, rownames=['Actual'], colnames=['Predicted'])
#print (crosstab)
print ("Total datatpoints in datasets : \t"+str(crosstab['0'][0]))
print ("No of correctly predicted datasets :\t"+str(crosstab['0'][1]))
correct=crosstab['0']['1']#+crosstab['0']['0']
total=crosstab['0']['0']#+crosstab['0']['0']
accuracy=float(correct)/float(total)
print ("Accuracy of the KNN model :\t\t{}%".format(accuracy*100))
