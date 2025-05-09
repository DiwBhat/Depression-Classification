import pytest 
import pandas as pd

import app as streamlit_app 
import recommendation_logic

@pytest.fixture
def sample_original_data_fixture():
    # Data for all_students_original_values
    return pd.DataFrame({
        'Age': [20, 22, 25, 21, 23],
        'Academic Pressure': [2, 4, 3, 5, 1],
        'Sleep_Ordinal': [2, 0, 1, 3, 2], 
        'Financial Stress': [1, 3, 2, 4, 1],
        'Study Satisfaction': [4, 2, 3, 1, 5],
        'Work/Study Hours': [6, 10, 8, 12, 5],
        'Depression': [0, 1, 0, 1, 0] 
    })

@pytest.fixture
def mock_expected_features_for_similarity_fixture():
    # Simplified feature list for similarity tests
    return [
        'Age', 'Academic Pressure', 'Sleep_Ordinal', 'Financial Stress', 
        'Study Satisfaction', 'Work/Study Hours', 'Gender_Male', 'Region_Western'
    ]

@pytest.fixture
def sample_processed_data_fixture(mock_expected_features_for_similarity_fixture, sample_original_data_fixture):
    # Data for all_students_features_processed
    df = pd.DataFrame({
        'Age': [0.2, 0.4, 0.6, 0.3, 0.5], 
        'Academic Pressure': [0.25, 0.75, 0.5, 1.0, 0.0],
        'Sleep_Ordinal': [0.66, 0.0, 0.33, 1.0, 0.66],
        'Financial Stress': [0.0, 0.5, 0.25, 0.75, 0.0],
        'Study Satisfaction': [0.75, 0.25, 0.5, 0.0, 1.0],
        'Work/Study Hours': [0.2, 0.6, 0.4, 0.8, 0.1],
        'Gender_Male': [1, 0, 1, 1, 0],
        'Region_Western': [0, 1, 1, 0, 0]
    }, columns=mock_expected_features_for_similarity_fixture)
    df.index = sample_original_data_fixture.index 
    return df

@pytest.fixture
def mock_expected_features_app_fixture():
    features = [
        'Age', 'Academic Pressure', 'CGPA', 'Study Satisfaction', 'Suicidal_Thoughts', 
        'Work/Study Hours', 'Financial Stress', 'Family History of Mental Illness', 
        'Sleep_Ordinal', 'Total_Stress', 'Gender_Male', 

        'Region_Northern', 'Region_Western', 'Region_Southern', 'Region_Eastern',
        'Dietary Habits_Moderate', 'Dietary Habits_Unhealthy', 'Dietary Habits_Others', 
        'Degree_Level_Class 12', 'Degree_Level_Masters', 'Degree_Level_PhD', 'Degree_Level_Other/Unknown',
        'Degree_Field_Engineering/Tech', 'Degree_Field_Medical/Pharma',
        'Degree_Field_Arts/Humanities/Edu', 'Degree_Field_Business/Mgmt', 'Degree_Field_General/Other',
        'Degree_Field_Law', 'Degree_Field_PhD', 'Degree_Field_Science', 
        'Degree_Type_Science/Tech', 'Degree_Type_Non-Science/Arts/Business/Law',
        'Age_Group_25-30', 'Age_Group_31-39' 
    ]
    # Add dummy city columns 
    for city in streamlit_app.valid_cities:
        if city != streamlit_app.DROPPED_CITY and f"City_{city}" not in features:
            features.append(f"City_{city}")
    # Add dummy degree columns 
    for deg in streamlit_app.valid_degrees:
        # Sanitize degree 
        col_deg_name = f"Degree_{deg.replace('.', '').replace(' ', '_').replace('/', '_')}"
        if deg != streamlit_app.DROPPED_DEGREE and col_deg_name not in features:
             features.append(col_deg_name)
    return sorted(list(set(features)))


# Tests for recommendation_logic.py
def test_get_risk_level():
    assert recommendation_logic.get_risk_level(0.1) == "Low"
    assert recommendation_logic.get_risk_level(0.39) == "Low"
    assert recommendation_logic.get_risk_level(0.4) == "Medium"
    assert recommendation_logic.get_risk_level(0.69) == "Medium"
    assert recommendation_logic.get_risk_level(0.7) == "High"
    assert recommendation_logic.get_risk_level(0.9) == "High"

def test_find_similar_students_basic(mock_expected_features_for_similarity_fixture, sample_processed_data_fixture):
    user_features_df = pd.DataFrame([{
        'Age': 0.25, 'Academic Pressure': 0.3, 'Sleep_Ordinal': 0.7,
        'Financial Stress': 0.1, 'Study Satisfaction': 0.8, 'Work/Study Hours': 0.25,
        'Gender_Male': 1, 'Region_Western': 0
    }], columns=mock_expected_features_for_similarity_fixture)

    similar_indices = recommendation_logic.find_similar_students(
        user_features_df, 
        sample_processed_data_fixture, 
        top_n=2
    )
    assert len(similar_indices) == 2
    assert isinstance(similar_indices, pd.Index)

def test_find_similar_students_no_common_features(sample_processed_data_fixture):
    user_features_df = pd.DataFrame([{'Some_Other_Feature': 1}])
    similar_indices = recommendation_logic.find_similar_students(
        user_features_df, sample_processed_data_fixture, top_n=2
    )
    assert len(similar_indices) == 0
        
def test_find_similar_students_scaling(mock_expected_features_for_similarity_fixture):
    user_features_df = pd.DataFrame([{
        'Age': 22, 'Academic Pressure': 3, 'Sleep_Ordinal': 1, 
        'Financial Stress': 2, 'Study Satisfaction': 3, 'Work/Study Hours': 7,
        'Gender_Male': 0, 'Region_Western': 1
    }], columns=mock_expected_features_for_similarity_fixture) 

    dataset_for_scaling_test = pd.DataFrame({
        'Age': [20.0, 40.0], 'Academic Pressure': [1.0, 5.0], 'Sleep_Ordinal': [0.0, 3.0], # Ensure float for scaling
        'Financial Stress': [1.0, 5.0], 'Study Satisfaction': [1.0, 5.0], 'Work/Study Hours': [4.0, 12.0],
        'Gender_Male': [0.0, 1.0], 'Region_Western': [1.0, 0.0] # Make OHE float too for consistency
    }, columns=mock_expected_features_for_similarity_fixture).astype(float)


    similar_indices = recommendation_logic.find_similar_students(
        user_features_df.astype(float), 
        dataset_for_scaling_test, 
        top_n=1
    )
    assert len(similar_indices) == 1

def test_generate_direct_recommendations():
    user_input = {
        'Academic Pressure': 5, 
        'Sleep Duration': 'Less than 5 hours', 
        'Financial Stress': 1, 'Study Satisfaction': 4, 'Work/Study Hours': 6
    }
    user_proba = 0.2 
    
    recommendations = recommendation_logic.generate_recommendations(
        user_input, None, user_proba, None, None 
    )
    
    categories_found = [r['category'] for r in recommendations]
    assert "High Academic Pressure" in categories_found
    assert "Poor Sleep Patterns" in categories_found
    if not any(c in ["High Academic Pressure", "Poor Sleep Patterns"] for c in categories_found):
        assert any("General" in c for c in categories_found)

# Tests for app.py logic 
# Note: Testing functions from app.py, not the Streamlit UI interactions.

def test_get_degree_level_app():
    assert streamlit_app.get_degree_level("Class 12") == "Class 12"
    assert streamlit_app.get_degree_level("PhD") == "PhD"
    assert streamlit_app.get_degree_level("M.Tech") == "Masters"
    assert streamlit_app.get_degree_level("BSc") == "Bachelors"

def test_get_degree_field_app():
    assert streamlit_app.get_degree_field("M.Tech") == "Engineering/Tech"
    assert streamlit_app.get_degree_field("MBBS") == "Medical/Pharma"

def test_get_degree_type_app():
    assert streamlit_app.get_degree_type("Engineering/Tech") == "Science/Tech"
    assert streamlit_app.get_degree_type("Law") == "Non-Science/Arts/Business/Law"

def test_preprocess_input_streamlit_basic(mock_expected_features_app_fixture, monkeypatch):
    monkeypatch.setattr(streamlit_app, 'DROPPED_GENDER', 'Female')
    monkeypatch.setattr(streamlit_app, 'DROPPED_REGION', 'Central') 
    monkeypatch.setattr(streamlit_app, 'DROPPED_DIET', 'Healthy')   
    monkeypatch.setattr(streamlit_app, 'DROPPED_DEGREE_LEVEL', 'Bachelors') 
    monkeypatch.setattr(streamlit_app, 'DROPPED_DEGREE_FIELD', 'Architecture') 
    monkeypatch.setattr(streamlit_app, 'DROPPED_DEGREE_TYPE', 'General/Other')  
    monkeypatch.setattr(streamlit_app, 'DROPPED_AGE_GROUP', '18-24')  
    
    user_input = {
        'Age': 22, 'Gender': 'Male', 'City': 'Pune', 
        'Academic Pressure': 3, 'CGPA': 8.0, 'Study Satisfaction': 4,
        'Sleep Duration': '7-8 hours', 'Dietary Habits': 'Moderate',
        'Degree': 'B.Tech', 'Suicidal Thoughts': 'No', 'Work/Study Hours': 8,
        'Financial Stress': 2, 'Family History': 'No'
    }
    
    current_mock_features = mock_expected_features_app_fixture[:]

    expected_active_flags = {
        "Gender_Male": 1.0,
        "Region_Western": 1.0, # 
        "Dietary Habits_Moderate": 1.0,
        "Degree_Level_Bachelors": 1.0, 
        "Degree_Field_Engineering/Tech": 1.0,
        "Degree_Type_Science/Tech": 1.0,
        "Age_Group_18-24": 1.0 
    }

    processed_df = streamlit_app.preprocess_input_streamlit(user_input, current_mock_features)
    
    assert processed_df is not None
    assert processed_df.shape == (1, len(current_mock_features))
    assert processed_df.loc[0, 'Age'] == 22.0
    assert processed_df.loc[0, 'Total_Stress'] == 5.0 
    assert processed_df.loc[0, 'Sleep_Ordinal'] == 2.0 

    for flag_col, expected_val in expected_active_flags.items():
        if flag_col in current_mock_features: 
            assert processed_df.loc[0, flag_col] == expected_val, f"Mismatch for {flag_col}"
        elif expected_val == 1.0: 
            pass 