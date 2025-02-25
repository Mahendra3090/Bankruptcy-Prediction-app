import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
#from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# Load dataset 
def load_dataset():
    data = pd.read_csv("C:/Users/Asus/Desktop/ML_Project/data/Bankruptcy1_prevention.csv", delimiter=";")
    print(data.head())
    # Strip column names (removes any extra spaces)
    data.columns = data.columns.str.strip()


    return data

def create_model(data):
    # Identify duplicate rows and assign weights
    duplicate_mask = data.duplicated(keep=False)
    weights = np.ones(len(data))
    weights[duplicate_mask] = 0.5  # Reduce weight for duplicat
    # Convert categorical labels to numeric values
    data['class'] = data['class'].map({'bankruptcy': 1, 'non-bankruptcy': 0})

    # Ensure target variable is numeric
    data['class'] = data['class'].astype(int)

    # Split features and target
    X = data.drop(columns=['class'])
    y = data['class']

    # One-hot encode categorical features if needed
    #encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    #X_encoded = encoder.fit_transform(X.select_dtypes(include=['object']))
    #X_numeric = X.select_dtypes(exclude=['object']).values
    #X = np.hstack((X_numeric, X_encoded))

    # Handle class imbalance without using sample_weight in SMOTE
    #smote = SMOTE(random_state=42)
    #X_resampled, y_resampled = smote.fit_resample(X, y)

    # Train-test split
    #X_train, X_test, y_train, y_test = train_test_split(
    #X_resampled, y_resampled, test_size=0.2, random_state=42)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

    # Feature scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train Random Forest model with weights manually applied
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_train, y_train)

    # Train XGBoost model with weights manually applied
    xgb_model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
    xgb_model.fit(X_train, y_train)

    #test model(Rendom Forest)
    y_pred = rf_model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    #test model(xbg_model)
    y_pred = xgb_model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

 


    return rf_model,xgb_model,smote,scaler
    
 
def main():
    data = load_dataset()

    rf_model, xgb_model,smote, scaler = create_model(data)

    with open('model/rf_model.pkl', 'wb') as f:
        pickle.dump(rf_model,f)
    
    with open('model/xgb_model.pkl', 'wb') as f:
        pickle.dump(xgb_model,f)

    #with open('model/encoder.pkl', 'wb') as f:
    #    pickle.dump(encoder,f)
    
    #with open('model/smote.pkl', 'wb') as f:
    #    pickle.dump(smote,f)

    with open('model/scaler.pkl','wb') as f:
        pickle.dump(scaler,f)






if __name__ == '__main__':
    main()
