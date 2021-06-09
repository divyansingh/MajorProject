from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import re
import string


# load_model=pickle.load(open('model.sav','rb'))
app = Flask(__name__)



@app.route('/')
def man():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def home():
    data1 = request.form['a']
   
    # arr = np.array([[data1]])
    # arr = np.array([[data1]])
    # load_model=pickle.load(open("model.sav",'rb'))
    # pred = load_model(arr)

    


    df_fake = pd.read_csv("Fake.csv")
    df_true = pd.read_csv("True.csv")

    df_fake["class"] = 0
    df_true["class"] = 1

    df_fake_manual_testing = df_fake.tail(10)
    for i in range(23480,23470,-1):
        df_fake.drop([i], axis = 0, inplace = True)
    df_true_manual_testing = df_true.tail(10)
    for i in range(21416,21406,-1):
        df_true.drop([i], axis = 0, inplace = True)



    df_fake_manual_testing["class"] = 0
    df_true_manual_testing["class"] = 1


    df_manual_testing = pd.concat([df_fake_manual_testing,df_true_manual_testing], axis = 0)
    df_manual_testing.to_csv("manual_testing.csv")



    df_marge = pd.concat([df_fake, df_true], axis =0 )
    df_marge.head(10)

    df_marge.columns

    df = df_marge.drop(["title", "subject","date"], axis = 1)


    df.isnull().sum()

    df = df.sample(frac = 1)

    df.reset_index(inplace = True)
    df.drop(["index"], axis = 1, inplace = True)

    def wordopt(text):
        text = text.lower()
        text = re.sub('\[.*?\]', '', text)
        text = re.sub("\\W"," ",text) 
        text = re.sub('https?://\S+|www\.\S+', '', text)
        text = re.sub('<.*?>+', '', text)
        text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
        text = re.sub('\n', '', text)
        text = re.sub('\w*\d\w*', '', text)    
        return text


    df["text"] = df["text"].apply(wordopt)

    x = df["text"]
    y = df["class"]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

    from sklearn.feature_extraction.text import TfidfVectorizer


    vectorization = TfidfVectorizer()
    xv_train = vectorization.fit_transform(x_train)
    xv_test = vectorization.transform(x_test)


    from sklearn.linear_model import LogisticRegression

    LR = LogisticRegression()
    LR.fit(xv_train,y_train)

    pred_lr=LR.predict(xv_test)

    from sklearn.tree import DecisionTreeClassifier

    DT = DecisionTreeClassifier()
    DT.fit(xv_train, y_train)

    pred_dt = DT.predict(xv_test)


    from sklearn.ensemble import RandomForestClassifier

    RFC = RandomForestClassifier(random_state=0)
    RFC.fit(xv_train, y_train)


    pred_rfc = RFC.predict(xv_test)

    def output_lable(n):
        if n == 0:
            return "Fake News"
        elif n == 1:
            return "Not A Fake News"
        
    def output_lable(n):
        if n == 0:
            return "Fake News"
        elif n == 1:
            return "Not A Fake News"
        
    def manual_testing(news):
        testing_news = {"text":[news]}
        new_def_test = pd.DataFrame(testing_news)
        new_def_test["text"] = new_def_test["text"].apply(wordopt) 
        new_x_test = new_def_test["text"]
        new_xv_test = vectorization.transform(new_x_test)
        pred_LR = LR.predict(new_xv_test)
        pred_DT = DT.predict(new_xv_test)
    
        pred_RFC = RFC.predict(new_xv_test)

        return ("\n\nLR Prediction: {} \nDT Prediction: {}  \nRFC Prediction: {}".format(output_lable(pred_LR[0]),output_lable(pred_DT[0]), output_lable(pred_RFC[0])))

    data=manual_testing(data1)













    if request.method == 'POST':
        model.save()
        # Failure to return a redirect or render_template
    else:
        return render_template('after.html',data)



if __name__ == "__main__":
    app.run(debug=True)















