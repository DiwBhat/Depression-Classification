import streamlit as st
import pandas as pd
import joblib
import xgboost as xgb
import re 
import warnings

import recommendation_logic 

st.set_page_config(layout="wide")
warnings.filterwarnings('ignore') 

# Configuration
# Defining categories that were potentially dropped by drop_first=True during OHE
DROPPED_GENDER = 'Female' 
DROPPED_CITY = 'Agra'     
DROPPED_DIET = 'Healthy'  
DROPPED_DEGREE = 'BA'     
DROPPED_AGE_GROUP = '18-24' 
DROPPED_REGION = 'Central'   
DROPPED_DEGREE_LEVEL = 'Bachelors' 
DROPPED_DEGREE_FIELD = 'Architecture' 
DROPPED_DEGREE_TYPE = 'General/Other' 

# Mappings used during preprocessing

sleep_map = recommendation_logic.sleep_map 
DEFAULT_SLEEP_ORDINAL = recommendation_logic.DEFAULT_SLEEP_ORDINAL

# Bins/Labels used for Age_Group 
age_bins = recommendation_logic.age_bins
age_labels = recommendation_logic.age_labels


# Region mapping 
region_map = {
    'Delhi': 'Northern', 'Srinagar': 'Northern', 'Ludhiana': 'Northern',
    'Chandigarh': 'Northern', 'Jaipur': 'Northern', 'Ghaziabad': 'Northern',
    'Faridabad': 'Northern', 'Meerut': 'Northern', 'Kanpur': 'Northern',
    'Lucknow': 'Northern', 'Varanasi': 'Northern', 'Agra': 'Northern',
    'Bhopal': 'Central', 'Indore': 'Central', 'Nagpur': 'Central', 'Raipur': 'Central',
    'Mumbai': 'Western', 'Pune': 'Western', 'Ahmedabad': 'Western',
    'Surat': 'Western', 'Vadodara': 'Western', 'Rajkot': 'Western',
    'Thane': 'Western', 'Nashik': 'Western', 'Kalyan': 'Western',
    'Vasai-Virar': 'Western',
    'Bangalore': 'Southern', 'Chennai': 'Southern', 'Hyderabad': 'Southern',
    'Visakhapatnam': 'Southern', 'Coimbatore': 'Southern', 'Kochi': 'Southern',
    'Kolkata': 'Eastern', 'Patna': 'Eastern', 'Ranchi': 'Eastern', 'Bhubaneswar': 'Eastern',
    'Guwahati': 'Northeastern'
}

valid_cities = sorted(list(region_map.keys())) 

# Degree lists for dropdowns 
valid_degrees = sorted(['B.Com', 'B.Ed', 'B.Pharm', 'B.Tech', 'BA', 'BBA', 'BCA', 'BE', 
                        'BHM', 'BSc', 'Class 12', 'LLB', 'LLM', 'M.Com', 'M.Ed', 
                        'M.Pharm', 'M.Tech', 'MA', 'MBA', 'MBBS', 'MCA', 'MD', 
                        'ME', 'MHM', 'MSc', 'Others', 'PhD'])

# Load Model and Columns
@st.cache_resource 
def load_model_and_cols(model_path, cols_path):
    """Loads the saved model and feature columns."""
    try:
        if model_path.endswith(".joblib"):
             model = joblib.load(model_path)
        # Fallback to XGBoost native load if joblib fails or different format used
        elif model_path.endswith(".json") or model_path.endswith(".xgb"):
             model = xgb.XGBClassifier() # Initialize empty model
             model.load_model(model_path) # Load native format
        else:
             st.error(f"Unsupported model file format: {model_path}")
             return None, None
             
        feature_cols = joblib.load(cols_path)
        print("Model and columns loaded successfully.") 
        return model, feature_cols
    except FileNotFoundError:
        st.error(f"Error: Model ('{model_path}') or Columns ('{cols_path}') file not found.")
        return None, None
    except Exception as e:
        st.error(f"Error loading model or columns: {e}")
        return None, None

model_file = 'Models/xgboost_tuned_model.joblib'
cols_file = 'Models/feature_columns_final.joblib'
model, expected_features = load_model_and_cols(model_file, cols_file)

# Load Full Dataset for Recommendations 
@st.cache_data 
def load_and_prepare_datasets(cleaned_data_path, all_feature_names_from_model):
    try:
        full_df_original_values = pd.read_csv(cleaned_data_path)
        print(f"DEBUG: Full original dataset loaded. Shape: {full_df_original_values.shape}")

        all_students_features_processed = full_df_original_values[all_feature_names_from_model].copy()
        print(f"DEBUG: all_students_features_processed shape: {all_students_features_processed.shape}")
        
        return all_students_features_processed, full_df_original_values 
    except FileNotFoundError:
        st.error(f"Error: Full dataset '{cleaned_data_path}' not found for recommendations.")
        return None, None
    except Exception as e:
        st.error(f"Error loading/preparing full dataset: {e}")
        return None, None

all_students_features_processed_global, all_students_original_values_global = None, None
if expected_features: 
    all_students_features_processed_global, all_students_original_values_global = load_and_prepare_datasets(
        'cleaned_student_data.csv', 
        expected_features 
    )

# Helper functions for Degree Parsing 

def get_degree_level(degree):
    degree = str(degree).strip()
    if degree == 'Class 12': 
        return 'Class 12'
    if degree == 'PhD': 
        return 'PhD'
    if degree == 'Others' or degree == 'nan': 
        return 'Other/Unknown'
    if degree.startswith(('M.', 'ME', 'MD', 'MS', 'LLM', 'MA', 'MBA', 'MCA', 'MHM', 'MSc')): 
        return 'Masters'
    if degree.startswith(('B.', 'BE', 'BA', 'BBA', 'BCA', 'BHM', 'BSc', 'LLB')): 
        return 'Bachelors'
    return 'Other/Unknown'

def get_degree_field(degree):
    degree = str(degree).strip().upper()
    if degree in ['CLASS 12', 'OTHERS', 'NAN']: 
        return 'General/Other'
    if re.search(r'B\.?TECH|BE|ME|M\.?TECH|BCA|MCA', degree): 
        return 'Engineering/Tech'
    if re.search(r'MBBS|MD|B\.?PHARM|M\.?PHARM|BDS', degree): 
        return 'Medical/Pharma'
    if re.search(r'LLB|LLM', degree): 
        return 'Law'
    if re.search(r'BBA|MBA|B\.?COM|M\.?COM|BHM|MHM', degree): 
        return 'Business/Mgmt'
    if re.search(r'BA|MA|B\.?ED|M\.?ED', degree): 
        return 'Arts/Humanities/Edu'
    if re.search(r'BSC|MSC', degree): 
        return 'Science'
    if re.search(r'B\.?ARCH', degree): 
        return 'Architecture'
    if degree == 'PHD': 
        return 'PhD'
    return 'General/Other'

def get_degree_type(degree_field):
    science_tech_fields = ['Engineering/Tech', 'Medical/Pharma', 'Science', 'Architecture', 'PhD']
    if degree_field in science_tech_fields: 
        return 'Science/Tech'
    if degree_field == 'General/Other': 
        return 'General/Other'
    return 'Non-Science/Arts/Business/Law'

# Preprocessing Function
def preprocess_input_streamlit(user_input_dict, feature_names):
    """
    Preprocesses raw user input dictionary into a DataFrame suitable for the model.
    Args:
        user_input_dict (dict): Raw inputs from Streamlit widgets.
        feature_names (list): List of expected feature columns in order.
    Returns:
        pd.DataFrame: Single-row DataFrame ready for prediction, or None if error.
    """
    try:
        df_input = pd.DataFrame(0.0, index=[0], columns=feature_names)
        # Basic Mappings & Calculations 
        input_data = {}
        input_data['Age'] = float(user_input_dict.get('Age', 0))
        input_data['Academic Pressure'] = float(user_input_dict.get('Academic Pressure', 0))
        input_data['CGPA'] = float(user_input_dict.get('CGPA', 0))
        input_data['Study Satisfaction'] = float(user_input_dict.get('Study Satisfaction', 0))
        input_data['Work/Study Hours'] = float(user_input_dict.get('Work/Study Hours', 0))
        input_data['Financial Stress'] = float(user_input_dict.get('Financial Stress', 0))

        input_data['Suicidal_Thoughts'] = 1.0 if user_input_dict.get('Suicidal Thoughts', 'No') == 'Yes' else 0.0
        input_data['Family History of Mental Illness'] = 1.0 if user_input_dict.get('Family History', 'No') == 'Yes' else 0.0
        
        sleep_str = user_input_dict.get('Sleep Duration', '7-8 hours')
        input_data['Sleep_Ordinal'] = float(sleep_map.get(sleep_str, DEFAULT_SLEEP_ORDINAL))
        
        input_data['Total_Stress'] = input_data['Academic Pressure'] + input_data['Financial Stress']

        # Populate Non-One-Hot Columns in DataFrame 
        for key, value in input_data.items():
            if key in df_input.columns:
                df_input.loc[0, key] = value
            else:
                 print(f"Debug: Column '{key}' from input_data not in expected_features.") 

        # Handle One-Hot Encoded Columns 
        
        # Gender (Handle dropped category)
        if user_input_dict.get('Gender') != DROPPED_GENDER:
            gender_col = f"Gender_{user_input_dict.get('Gender')}"
            if gender_col in df_input.columns:
                df_input.loc[0, gender_col] = 1.0

        # Region (Handle dropped category)
        user_city = user_input_dict.get('City')
        user_region = region_map.get(user_city, 'Other/Unknown') 
        if user_region != DROPPED_REGION:
            region_col = f"Region_{user_region}"
            if region_col in df_input.columns:
                df_input.loc[0, region_col] = 1.0
            elif user_region != 'Other/Unknown':
                 print(f"Warning: Region column '{region_col}' not found in model features.")

        # Dietary Habits (Handle dropped category)
        user_diet = user_input_dict.get('Dietary Habits')
        if user_diet != DROPPED_DIET:
            diet_col = f"Dietary Habits_{user_diet}"
            if diet_col in df_input.columns:
                df_input.loc[0, diet_col] = 1.0
            elif user_diet != 'Others' and user_diet is not None: # Added check for None
                 print(f"Warning: Diet column '{diet_col}' not found.")

        # Degree Generalization Features (Handle dropped categories)
        user_degree = user_input_dict.get('Degree')

        # Level
        level = get_degree_level(user_degree) 
        if level != DROPPED_DEGREE_LEVEL:
             level_col = f"Degree_Level_{level.replace('/', '_')}" 
             if level_col in df_input.columns:
                 df_input.loc[0, level_col] = 1.0
             elif level != 'Other/Unknown':
                 print(f"Warning: Degree Level column '{level_col}' not found.")
                 
        # Field
        field = get_degree_field(user_degree) 
        if field != DROPPED_DEGREE_FIELD:
            field_col = f"Degree_Field_{field}" 
            if field_col in df_input.columns:
                 df_input.loc[0, field_col] = 1.0
            elif field != 'General/Other' and field != 'PhD':
                 print(f"Warning: Degree Field column '{field_col}' not found.")

        # Type
        dtype = get_degree_type(field)
        if dtype != DROPPED_DEGREE_TYPE:
             type_col = f"Degree_Type_{dtype}" 
             if type_col in df_input.columns:
                 df_input.loc[0, type_col] = 1.0
             elif dtype != 'General/Other':
                  print(f"Warning: Degree Type column '{type_col}' not found.")

        # Age Group (Handle dropped category)
        age_group_str = pd.cut([input_data['Age']], bins=age_bins, labels=age_labels, right=True)[0]
        if age_group_str != DROPPED_AGE_GROUP:
            age_group_col = f"Age_Group_{age_group_str}"
            if age_group_col in df_input.columns:
                df_input.loc[0, age_group_col] = 1.0
            
        # Final check for missing columns 
        missing_cols = [col for col in feature_names if col not in df_input.columns]
        if missing_cols:
            print(f"Error: Columns missing in final input df: {missing_cols}")
            return None
            
        df_input = df_input[feature_names]

        return df_input

    except Exception as e:
        st.error(f"Error during preprocessing: {e}")
        import traceback
        traceback.print_exc() 
        return None

#  Streamlit App Layout 
st.title("Student Depression Prediction Model")

st.markdown("""
    **Disclaimer:** This tool provides a prediction based on a machine learning model trained on student data. 
    It is **NOT** a substitute for professional medical advice, diagnosis, or treatment. 
    If you are concerned about your mental health, please consult a qualified healthcare provider.
""")

st.divider()

if model is None or expected_features is None:
    st.error("Model could not be loaded. Application cannot proceed.")
else:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Demographics & Academics")
        age = st.number_input("Age", min_value=17, max_value=40, value=25, step=1)
        gender = st.selectbox("Gender", ['Female', 'Male']) 
        city = st.selectbox("City", valid_cities, index=valid_cities.index('Pune') if 'Pune' in valid_cities else 0) # Default Pune
        degree = st.selectbox("Highest Current/Completed Degree", valid_degrees, index=valid_degrees.index('B.Tech') if 'B.Tech' in valid_degrees else 0)
        cgpa = st.number_input("CGPA (Cumulative Grade Point Average)", min_value=0.0, max_value=10.0, value=8.0, step=0.1)
        
    with col2:
        st.subheader("Well-being & Stressors")
        sleep = st.selectbox("Average Sleep Duration per Night", list(sleep_map.keys()), index=2) # Default 7-8
        diet = st.selectbox("Dietary Habits", ['Healthy', 'Moderate', 'Unhealthy', 'Others'], index=1)
        work_study_hours = st.slider("Average Work/Study Hours per Day", min_value=0, max_value=16, value=8, step=1)
        academic_pressure = st.slider("Academic Pressure Level (1-5)", min_value=1, max_value=5, value=3, step=1)
        financial_stress = st.slider("Financial Stress Level (1-5)", min_value=1, max_value=5, value=2, step=1)
        study_satisfaction = st.slider("Study Satisfaction Level (1-5)", min_value=1, max_value=5, value=3, step=1)
        suicidal = st.radio("Have you ever had suicidal thoughts?", ('No', 'Yes'))
        family_history = st.radio("Family History of Mental Illness?", ('No', 'Yes'))

    st.divider()

    # Prediction Button
    if st.button("Predict Depression Likelihood", type="primary"):
        user_data = {
            'Age': age,
            'Gender': gender if gender != 'Other' else DROPPED_GENDER, 
            'City': city,
            'Academic Pressure': academic_pressure,
            'CGPA': cgpa,
            'Study Satisfaction': study_satisfaction,
            'Sleep Duration': sleep,
            'Dietary Habits': diet,
            'Degree': degree,
            'Suicidal Thoughts': suicidal,
            'Work/Study Hours': work_study_hours,
            'Financial Stress': financial_stress,
            'Family History': family_history,
        }

        # Preprocess the input
        processed_df = preprocess_input_streamlit(user_data, expected_features)

        if processed_df is not None:
            # Make prediction
            try:
                prediction_proba = model.predict_proba(processed_df)[0] 
                probability_depression = prediction_proba[1] 
                
                prediction_threshold = 0.4
                prediction_class = 1 if probability_depression >= prediction_threshold else 0

                # Display Prediction result
                st.subheader("Prediction Result")
                if prediction_class == 1:
                    st.warning(f"**Prediction: Likely Depressed** (Probability: {probability_depression:.2f})")
                    st.markdown(f"""
                        Based on the input provided, the model predicts a higher likelihood of depression (probability >= {prediction_threshold}). 
                        **Note**, this is a statistical prediction, not a diagnosis. Please reach out to a mental health professional 
                        or trusted person if you are feeling distressed.
                    """)
                else:
                    st.success(f"**Prediction: Likely Not Depressed** (Probability of Depression: {probability_depression:.2f})")
                    st.markdown(f"""
                        Based on the input provided, the model predicts a lower likelihood of depression (probability < {prediction_threshold}). 
                        Continue prioritizing your mental well-being. If you ever feel overwhelmed, seeking support is always a good option.
                    """)

                st.subheader("Personalized Recommendations")
                recommendations = recommendation_logic.generate_recommendations(
                    user_input_dict=user_data,
                    user_processed_features=processed_df, 
                    user_prediction_proba=probability_depression,
                    all_students_features_processed=all_students_features_processed_global,
                    all_students_original_values=all_students_original_values_global
                )

                if recommendations:
                    for i, rec_item in enumerate(recommendations):
                        with st.expander(f"**{i+1}. Regarding: {rec_item['category']}**", expanded=(i==0)): # Expand first one
                            for suggestion in rec_item['suggestions']:
                                st.markdown(f"- {suggestion}")
                else:
                    st.info("No specific recommendations triggered, but maintaining a balanced lifestyle is always beneficial.")
                
                st.divider()
                st.markdown("If you are feeling distressed or need support, please consider reaching out to a mental health professional, a trusted friend or family member, or campus counseling services.")

            except Exception as e:
                st.error(f"Error during prediction: {e}")
        else:
            st.error("Could not preprocess the input data. Please check values.")