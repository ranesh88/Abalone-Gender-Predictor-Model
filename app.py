from flask import Flask, render_template, request
from flask_cors import CORS,cross_origin
from sklearn.preprocessing import StandardScaler
import pickle
import pandas as pd
import numpy as np

from warnings import filterwarnings
filterwarnings('ignore')

app = Flask(__name__)
cors=CORS(app)
# Load the sentiment analysis model and TF-IDF vectorizer
with open('lg_classifier.pkl', 'rb') as f:
    clf = pickle.load(f)
with open('preprocessor.pkl', 'rb') as f:
    preprocessor = pickle.load(f)
abalone=pd.read_csv('abalone_data.csv')

def predict(Length,Diameter,Height,Whole_weight,Shucked_weight,Viscera_weight,Shell_weight,Rings):
    # Prepare features array
    features = np.array([[Length,Diameter,Height,Whole_weight,Shucked_weight,Viscera_weight,Shell_weight,Rings]],dtype = 'object')

    # transformed featured
    transformed_features = preprocessor.transform(features)

    # predict by model
    result = clf.predict(transformed_features).reshape(1, -1)

    return result[0]

@app.route('/', methods=['GET', 'POST'])
@cross_origin()
def analyze_gender():
    if request.method == 'POST':
        Length = float(request.form.get('Length'))
        Diameter = float(request.form.get('Diameter'))
        Height = float(request.form.get('Height'))
        Whole_weight = float(request.form.get('Whole_weight'))
        Shucked_weight =float(request.form.get('Shucked_weight'))
        Viscera_weight = float(request.form.get('Viscera_weight'))
        Shell_weight = float(request.form.get('Shell_weight'))
        Rings =int(request.form.get('Rings'))

        prediction = predict(Length,Diameter,Height,Whole_weight,Shucked_weight,Viscera_weight,Shell_weight,Rings)

        return render_template('index.html', prediction=prediction)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True,port=8080,use_reloader=False)


