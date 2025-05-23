import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from joblib import dump, load
import os

# Function to train and save models
def train_and_save_models():
    # Load the dataset
    df = pd.read_csv("C:/Users/arpit/Social Media Tourism Analysis/Social Media Data for DSBA.csv")

    # Initial Preprocessing
    df['member_in_family'] = df['member_in_family'].replace('Three', 3)
    df['yearly_avg_Outstation_checkins'] = df['yearly_avg_Outstation_checkins'].replace('Three', 3)
    df['preferred_device'] = df['preferred_device'].replace(['iOS', 'ANDROID', 'Tab', 'Mobile', 'iOS and Android'], 'Mobile')
    df['preferred_device'] = df['preferred_device'].replace(['Laptop', 'laptop'], 'Laptop')

    # Split dataset into Laptop and Mobile
    le = LabelEncoder()
    df['preferred_device'] = df['preferred_device'].fillna('Mobile')
    df['preferred_device'] = le.fit_transform(df['preferred_device'])
    laptop_encoded = le.transform(['Laptop'])[0]
    mobile_encoded = le.transform(['Mobile'])[0]
    df_laptop = df[df['preferred_device'] == laptop_encoded].copy()
    df_mobile = df[df['preferred_device'] == mobile_encoded].copy()

    # Define columns
    numerical_cols = ['Yearly_avg_view_on_travel_page', 'total_likes_on_outstation_checkin_given', 
                     'yearly_avg_Outstation_checkins', 'member_in_family', 
                     'Yearly_avg_comment_on_travel_page', 'total_likes_on_outofstation_checkin_received',
                     'week_since_last_outstation_checkin', 'montly_avg_comment_on_company_page',
                     'travelling_network_rating', 'Adult_flag', 'Daily_Avg_mins_spend_on_traveling_page']
    categorical_cols = ['preferred_location_type', 'following_company_page', 'working_flag']

    # Selected features from previous analysis
    selected_features = {
        'Laptop': ['Yearly_avg_view_on_travel_page', 'total_likes_on_outofstation_checkin_received', 
                   'total_likes_on_outstation_checkin_given', 'yearly_avg_Outstation_checkins', 
                   'Daily_Avg_mins_spend_on_traveling_page'],
        'Mobile': ['total_likes_on_outofstation_checkin_received', 'total_likes_on_outstation_checkin_given', 
                   'Yearly_avg_view_on_travel_page', 'following_company_page', 'yearly_avg_Outstation_checkins']
    }

    models = {}
    for device, data in [('Laptop', df_laptop), ('Mobile', df_mobile)]:
        data = data.copy()
        # Preprocessing
        for col in numerical_cols:
            data[col] = pd.to_numeric(data[col], errors='coerce')
            data[col].fillna(data[col].median(), inplace=True)
        for col in categorical_cols:
            data[col].fillna(data[col].mode()[0], inplace=True)
        
        # Encode categorical variables
        le = LabelEncoder()
        data['Taken_product'] = le.fit_transform(data['Taken_product'].astype(str))
        for col in categorical_cols:
            data[col] = le.fit_transform(data[col])
        
        # Outlier treatment
        for col in numerical_cols:
            data[col] = data[col].clip(lower=data[col].quantile(0.05), upper=data[col].quantile(0.95))
        
        # Prepare data
        X = data[selected_features[device]]
        y = data['Taken_product']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=200 if device == 'Laptop' else 300)
        
        # Train fine-tuned model
        rf = RandomForestClassifier(n_estimators=200, max_depth=15, min_samples_split=5, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        
        # Save model
        dump(rf, f'{device}_model.joblib')
        models[device] = rf
        
        # Print performance
        y_pred = rf.predict(X_test)
        print(f"{device} Test F1: {f1_score(y_test, y_pred)}")

    return models, selected_features

# Check if models exist, else train and save
if not (os.path.exists('Laptop_model.joblib') and os.path.exists('Mobile_model.joblib')):
    models, selected_features = train_and_save_models()
else:
    models = {
        'Laptop': load('Laptop_model.joblib'),
        'Mobile': load('Mobile_model.joblib')
    }
    selected_features = {
        'Laptop': ['Yearly_avg_view_on_travel_page', 'total_likes_on_outofstation_checkin_received', 
                   'total_likes_on_outstation_checkin_given', 'yearly_avg_Outstation_checkins', 
                   'Daily_Avg_mins_spend_on_traveling_page'],
        'Mobile': ['total_likes_on_outofstation_checkin_received', 'total_likes_on_outstation_checkin_given', 
                   'Yearly_avg_view_on_travel_page', 'following_company_page', 'yearly_avg_Outstation_checkins']
    }

# Streamlit App
st.title("Social Media Tourism Prediction App")
st.write("Predict whether a user will purchase a travel product based on their social media behavior.")

# Device Selection
device = st.selectbox("Select Device", ["Laptop", "Mobile"])

# Input Fields
st.subheader(f"Enter Features for {device}")
inputs = {}
if device == 'Laptop':
    inputs['Yearly_avg_view_on_travel_page'] = st.number_input("Yearly Average Views on Travel Page", min_value=200.0, max_value=400.0, value=280.0)
    inputs['total_likes_on_outofstation_checkin_received'] = st.number_input("Total Likes on Out-of-Station Check-ins Received", min_value=2000, max_value=15000, value=6500)
    inputs['total_likes_on_outstation_checkin_given'] = st.number_input("Total Likes on Out-of-Station Check-ins Given", min_value=5000, max_value=50000, value=28000)
    inputs['yearly_avg_Outstation_checkins'] = st.number_input("Yearly Average Out-of-Station Check-ins", min_value=1, max_value=30, value=9)
    inputs['Daily_Avg_mins_spend_on_traveling_page'] = st.number_input("Daily Average Minutes Spent on Traveling Page", min_value=5, max_value=25, value=13)
else:
    inputs['total_likes_on_outofstation_checkin_received'] = st.number_input("Total Likes on Out-of-Station Check-ins Received", min_value=2000, max_value=10000, value=6500)
    inputs['total_likes_on_outstation_checkin_given'] = st.number_input("Total Likes on Out-of-Station Check-ins Given", min_value=5000, max_value=50000, value=28000)
    inputs['Yearly_avg_view_on_travel_page'] = st.number_input("Yearly Average Views on Travel Page", min_value=200.0, max_value=400.0, value=280.0)
    inputs['following_company_page'] = st.selectbox("Following Company Page", ['Yes', 'No'])
    inputs['yearly_avg_Outstation_checkins'] = st.number_input("Yearly Average Out-of-Station Check-ins", min_value=1, max_value=30, value=8)

# Preprocess Inputs
input_df = pd.DataFrame([inputs])
if device == 'Mobile':
    input_df['following_company_page'] = 1 if inputs['following_company_page'] == 'Yes' else 0
for col in selected_features[device]:
    if col in ['Yearly_avg_view_on_travel_page', 'total_likes_on_outstation_checkin_given', 
               'total_likes_on_outofstation_checkin_received', 'yearly_avg_Outstation_checkins', 
               'Daily_Avg_mins_spend_on_traveling_page']:
        input_df[col] = input_df[col].clip(lower=0, upper=input_df[col].quantile(0.95) if col in input_df else 50000)

# Prediction
if st.button("Predict"):
    model = models[device]
    prediction = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]
    result = 'Yes' if prediction == 1 else 'No'
    
    st.success(f"Prediction: User will {'purchase' if result == 'Yes' else 'not purchase'} a travel product.")
    st.write(f"Probability of Purchase: {prob:.2%}")

# Feature Importance
st.subheader("Feature Importance")
feature_importance = pd.DataFrame({
    'Feature': selected_features[device],
    'Importance': models[device].feature_importances_
}).sort_values(by='Importance', ascending=False)
fig, ax = plt.subplots()
sns.barplot(x='Importance', y='Feature', data=feature_importance, ax=ax)
plt.title(f'Feature Importance ({device})')
st.pyplot(fig)

# Cluster Insights
st.subheader("Cluster Insights")
if device == 'Laptop':
    st.write("""
    - Cluster 0: High views (276.97), moderate likes received (6552.52), high purchase rate (29.9%).
    - Cluster 1: Highest views (291.45), high likes given (44256.13), lowest purchase rate (15.9%).
    - Cluster 2: Moderate views (278.32), low likes received (5506.06), moderate purchase rate (27.6%).
    """)
else:
    st.write("""
    - Cluster 0: Moderate likes received (6532.91), moderate likes given (27667.35), moderate purchase rate (14.9%).
    - Cluster 1: High likes received (6698.72), high likes given (44110.49), low purchase rate (13.8%).
    - Cluster 2: Low likes received (6336.59), low likes given (11914.29), highest purchase rate (18.2%).
    """)

# Instructions
st.sidebar.header("Instructions")
st.sidebar.write("""
1. Select the device (Laptop or Mobile).
2. Enter the required feature values.
3. Click 'Predict' to see the prediction and probability.
4. View feature importance and cluster insights for context.
""")