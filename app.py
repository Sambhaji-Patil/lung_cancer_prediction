import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.model_selection import train_test_split
import joblib
from flask import Flask, render_template, request

app = Flask(__name__)

# Load and prepare data
df = pd.read_csv('survey_lung_cancer.csv')
Sex, Cancer = LabelEncoder(), LabelEncoder()
df['LUNG_CANCER'] = Cancer.fit_transform(df['LUNG_CANCER'])
df['GENDER'] = Sex.fit_transform(df['GENDER'])

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df.drop('LUNG_CANCER', axis=1), df['LUNG_CANCER'], test_size=0.3, random_state=19
)

def load_or_train_model(selected_model, problem_type):
    """
    Loads a saved model if available; otherwise, trains a new model.
    """
    model_file = f'{selected_model}_model.pkl'
    
    if os.path.exists(model_file):
        # Load the model if it exists
        model = joblib.load(model_file)
    else:
        # Train a new model if it doesn't exist
        if problem_type == 'regression':
            if selected_model == 'logistic_regression':
                model = LogisticRegression()
            elif selected_model == 'randomforest_regressor':
                model = RandomForestRegressor()
            elif selected_model == 'decisiontree_regressor':
                model = DecisionTreeRegressor()
            elif selected_model == 'svr':
                model = SVR()
        else:
            if selected_model == 'randomforest_classifier':
                model = RandomForestClassifier()
            elif selected_model == 'decisiontree_classifier':
                model = DecisionTreeClassifier()
            elif selected_model == 'svc':
                model = SVC(probability=True)
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Save the trained model
        joblib.dump(model, model_file)
    
    return model

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_percentage = 0  # Default when the page loads

    if request.method == 'POST':
        # Get user input
        problem_type = request.form['regression_classification']
        selected_model = request.form['model']
        gender = 1 if request.form['gender'] == 'Male' else 0
        age = int(request.form['age'])
        smoking = 2 if request.form['smoking'] == 'Yes' else 1
        yellow_fii = 2 if request.form['yellow_fii'] == 'Yes' else 1
        anxiety = 2 if request.form['anxiety'] == 'Yes' else 1
        peer_pres = 2 if request.form['peer_pres'] == 'Yes' else 1
        chronic_d = 2 if request.form['chronic_d'] == 'Yes' else 1
        fatigue = 2 if request.form['fatigue'] == 'Yes' else 1
        allergy = 2 if request.form['allergy'] == 'Yes' else 1
        wheezing = 2 if request.form['wheezing'] == 'Yes' else 1
        alcohol_c = 2 if request.form['alcohol_c'] == 'Yes' else 1
        coughing = 2 if request.form['coughing'] == 'Yes' else 1
        shortness_s = 2 if request.form['shortness_s'] == 'Yes' else 1
        swallowi = 2 if request.form['swallowi'] == 'Yes' else 1
        chest_pain = 2 if request.form['chest_pain'] == 'Yes' else 1

        # Prepare feature vector
        features = [[gender, age, smoking, yellow_fii, anxiety, peer_pres, chronic_d, fatigue, allergy, wheezing,
                     alcohol_c, coughing, shortness_s, swallowi, chest_pain]]  # Add other feature inputs

        # Load or train model
        model = load_or_train_model(selected_model, problem_type)

        # Prediction
        if problem_type == 'regression':
            prediction = model.predict(features)
            prediction_percentage = round(prediction[0] * 100, 3)
        else:
            prediction = model.predict_proba(features)[:, 1]
            prediction_percentage = round(prediction[0] * 100, 3)

        return render_template('index.html', prediction_percentage=prediction_percentage)
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
