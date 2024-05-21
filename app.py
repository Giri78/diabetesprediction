import streamlit as st
import pickle
import numpy as np
# try:
#     model = pickle.load(open('selected_features.pkl', 'rb'))
# except Exception as e:
#     st.error(f"Error loading the model: {e}")
# Load the trained model
model = pickle.load(open('model (2).pkl', 'rb'))

def predict_diabetes(age, hypertension, heart_disease, bmi, HbA1c_level, blood_glucose_level, smoking_history_encoded, gender_encoded):
    # Create input array
    features = np.array([age, hypertension, heart_disease, bmi, HbA1c_level, blood_glucose_level, smoking_history_encoded, gender_encoded]).reshape(1, -1)

    # Predict
    prediction = model.predict(features)

    return prediction[0]

def main():
    st.title('Diabetes Prediction')
    st.write('Fill out the form below to predict diabetes.')

    # Input form
    with st.form(key='diabetes_form'):
        age = st.number_input('Age', min_value=0, step=1)
        hypertension = st.radio('Hypertension', ['No', 'Yes'])
        heart_disease = st.radio('Heart Disease', ['No', 'Yes'])
        # bmi = st.number_input('BMI', min_value=0, step=0.1)
        bmi = st.number_input('BMI', min_value=0)

        HbA1c_level = st.selectbox('HbA1c Level', ['Normal', 'High'])
        blood_glucose_level = st.number_input('Blood Glucose Level', min_value=0, step=1)
        smoking_history_encoded = st.radio('Smoking History', ['No', 'Yes'])
        gender_encoded = st.radio('Gender', ['Female', 'Male'])

        submit_button = st.form_submit_button(label='Predict')

    # Convert categorical features to numeric
    hypertension = 1 if hypertension == 'Yes' else 0
    heart_disease = 1 if heart_disease == 'Yes' else 0
    smoking_history_encoded = 1 if smoking_history_encoded == 'Yes' else 0
    gender_encoded = 1 if gender_encoded == 'Male' else 0
    HbA1c_level = 1 if HbA1c_level == 'High' else 0

    # Perform prediction when form is submitted
    if submit_button:
        prediction = predict_diabetes(age, hypertension, heart_disease, bmi, HbA1c_level, blood_glucose_level, smoking_history_encoded, gender_encoded)

        if prediction == 1:
            st.error('You have diabetes!')
        else:
            st.success('You do not have diabetes.')

if __name__ == '__main__':
    main()







# import streamlit as st
# import pickle
# import numpy as np
# import pandas as pd

# try:
#     # Load the trained model
#     with open('diabetes.pkl', 'rb') as f:
#         model = pickle.load(f)
# except EOFError:
#     st.error("The pickle file appears to be empty or corrupted.")
#     model = None
# except Exception as e:
#     st.error(f"Error loading the model: {e}")
#     model = None

# def predict_diabetes(age, hypertension, heart_disease, bmi, HbA1c_level, blood_glucose_level, smoking_history, gender):
#     if model is None:
#         return None
    
#     # Create DataFrame with input features
#     data = {'age': [age],
#             'hypertension': [hypertension],
#             'heart_disease': [heart_disease],
#             'bmi': [bmi],
#             'HbA1c_level': [HbA1c_level],
#             'blood_glucose_level': [blood_glucose_level],
#             'smoking_history': [smoking_history],
#             'gender': [gender]}
    
#     df = pd.DataFrame(data)

#     # Predict
#     prediction = model.predict(df)

#     return prediction[0]

# def main():
#     st.title('Diabetes Prediction')
#     st.write('Fill out the form below to predict diabetes.')

#     # Input form
#     with st.form(key='diabetes_form'):
#         age = st.number_input('Age', min_value=0, step=1)
#         hypertension = st.radio('Hypertension', ['No', 'Yes'])
#         heart_disease = st.radio('Heart Disease', ['No', 'Yes'])
#         bmi = st.number_input('BMI', min_value=0)
#         HbA1c_level = st.selectbox('HbA1c Level', ['Normal', 'High'])
#         blood_glucose_level = st.number_input('Blood Glucose Level', min_value=0, step=1)
#         smoking_history = st.radio('Smoking History', ['No', 'Yes'])
#         gender = st.radio('Gender', ['Female', 'Male'])

#         submit_button = st.form_submit_button(label='Predict')

#     # Convert categorical features to numeric
#     hypertension_encoded = 1 if hypertension == 'Yes' else 0
#     heart_disease_encoded = 1 if heart_disease == 'Yes' else 0
#     smoking_history_encoded = 1 if smoking_history == 'Yes' else 0
#     gender_encoded = 1 if gender == 'Male' else 0
#     HbA1c_level_encoded = 1 if HbA1c_level == 'High' else 0

#     # Perform prediction when form is submitted
#     if submit_button:
#         prediction = predict_diabetes(age, hypertension_encoded, heart_disease_encoded, bmi, HbA1c_level_encoded, blood_glucose_level, smoking_history_encoded, gender_encoded)

#         if prediction is None:
#             st.error("Model is not available.")
#         elif prediction == 1:
#             st.error('You have diabetes!')
#         else:
#             st.success('You do not have diabetes.')

# if __name__ == '__main__':
#     main()
