from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

model_rf = pickle.load(open('heart_disease.pkl', 'rb'))
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
  
    features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']

    user_input = {feature: [float(request.form[feature])] for feature in features}
    user_DF = pd.DataFrame(user_input)

    pred_user = model_rf.predict(user_DF)

    return render_template('result.html', result=pred_user[0])

if __name__ == '__main__':
    app.run(debug=True)
