import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler 
from sklearn.metrics.pairwise import cosine_similarity 

# Configuration & Predefined Data 
sleep_map = {
    'Less than 5 hours': 0,
    '5-6 hours': 1,
    '7-8 hours': 2,
    'More than 8 hours': 3
}
DEFAULT_SLEEP_ORDINAL = 2 

# Inverse map for displaying human-readable sleep duration from ordinal
sleep_map_inv = {v: k for k, v in sleep_map.items()}

# Bins/Labels used for Age_Group 
age_bins = [17, 24, 30, 40] 
age_labels = ['18-24', '25-30', '31-39']

# Defining fields that are actionable and numeric/ordinal for comparison
COMPARISON_FIELDS = {
    'Academic Pressure': {'lower_is_better': True, 'display_name': 'Academic Pressure'},
    'Sleep_Ordinal': {'lower_is_better': False, 'display_name': 'Sleep Duration'}, # Higher ordinal is better
    'Financial Stress': {'lower_is_better': True, 'display_name': 'Financial Stress'},
    'Study Satisfaction': {'lower_is_better': False, 'display_name': 'Study Satisfaction'}, # Higher is better
    'Work/Study Hours': {'lower_is_better': True, 'display_name': 'Work/Study Hours'} #  Lower is better for stress
}

# Recommendation categories and suggestions
RECOMMENDATION_CATEGORIES = {
    "academic_pressure": {
        "high_threshold": 4,
        "suggestions": [
            "Take regular short study breaks (e.g., 5-10 minutes every hour).",
            "Try the Pomodoro Technique for focused study sessions.",
            "Break down large assignments into smaller, manageable tasks.",
            "Practice effective time management; plan your study schedule.",
            "Ensure you understand the material; seek help from professors or TAs if needed."
        ]
    },
    "sleep_patterns": {
        "low_threshold": 1,
        "suggestions": [
            "Aim for 7-9 hours of consistent sleep per night.",
            "Establish a regular sleep schedule (go to bed and wake up around the same time).",
            "Limit screen time (phones, laptops) at least 30-60 minutes before bed.",
            "Create a relaxing bedtime routine (e.g., reading, warm bath).",
            "Ensure your bedroom is dark, quiet, and cool."
        ]
    },
    "financial_stress": {
        "high_threshold": 4,
        "suggestions": [
            "Create a budget to track income and expenses.",
            "Look for part-time job opportunities or campus jobs if feasible.",
            "Explore scholarships, grants, or financial aid options.",
            "Consider workshops on financial literacy or counseling services offered by your institution."
        ]
    },
    "study_satisfaction": {
        "low_threshold": 2,
        "suggestions": [
            "Reflect on what aspects of your studies are causing dissatisfaction. Can they be changed?",
            "Connect with peers who are passionate about your field of study.",
            "Explore extracurricular activities or projects related to your interests.",
            "Talk to a career counselor or academic advisor about your study path.",
            "Ensure your study environment is conducive to learning and focus."
        ]
    },
    "work_study_hours": {
        "high_threshold_very_high": 12,
        "high_threshold_moderate": 10,
        "suggestions_very_high": [
            "Working/studying for such long hours can be unsustainable. Review your schedule for potential adjustments.",
            "Prioritize tasks and consider if any can be delegated or postponed.",
            "Ensure you are taking adequate breaks to avoid burnout."
        ],
        "suggestions_moderate": [
            "Long work/study hours can contribute to stress. Ensure you are balancing them with rest and leisure.",
            "Practice efficient study techniques to make the most of your time.",
            "Regularly assess if your workload is manageable."
        ]
    },
    "social_connection": {
        "suggestions": [
            "Make an effort to connect with friends, family, or classmates regularly.",
            "Join clubs or groups that align with your interests.",
            "Volunteer for a cause you care about."
        ]
    },
    "physical_activity": {
        "suggestions": [
            "Aim for at least 30 minutes of moderate physical activity most days of the week.",
            "Find an activity you enjoy, like walking, jogging, cycling, or team sports.",
            "Even short bursts of activity throughout the day can make a difference."
        ]
    },
    "mindfulness_stress_reduction": {
        "suggestions": [
            "Practice mindfulness meditation for a few minutes each day.",
            "Try deep breathing exercises to calm your nervous system.",
            "Engage in hobbies that you find relaxing and enjoyable."
        ]
    }
}

# Helper Functions 

def get_risk_level(probability_depression):
    """Categorizes depression risk based on probability."""
    if probability_depression >= 0.7:
        return "High"
    elif probability_depression >= 0.4:
        return "Medium"
    else:
        return "Low"

def find_similar_students(user_processed_features, all_students_features_processed, top_n=20):
    """
    Finds students in the dataset most similar to the user based on PREPROCESSED features.
    Args:
        user_processed_features (pd.DataFrame): Single-row DataFrame of the user's preprocessed features.
        all_students_features_processed (pd.DataFrame): DataFrame of ALL students' PREPROCESSED features.
        top_n (int): Number of similar students to return.
    Returns:
        pd.Index: Indices of the top_n most similar students from all_students_features_processed.
    """
    if user_processed_features is None or all_students_features_processed is None or all_students_features_processed.empty:
        print("Warning: Missing input for find_similar_students.")
        return pd.Index([])

    common_features = user_processed_features.columns.intersection(all_students_features_processed.columns)
    if len(common_features) == 0:
        print("Warning: No common features between user and dataset for similarity.")
        return pd.Index([])

    user_vec_raw = user_processed_features[common_features].astype(float)
    dataset_vecs_raw = all_students_features_processed[common_features].astype(float)

    potential_numerical_cols = user_vec_raw.select_dtypes(include=np.number).columns
    non_bool_numerical_cols = []
    for col in potential_numerical_cols:
    
        is_likely_ohe_or_binary = set(user_vec_raw[col].unique()).issubset({0, 1, 0.0, 1.0}) and user_vec_raw[col].nunique() <= 2
        if not is_likely_ohe_or_binary:
            non_bool_numerical_cols.append(col)
    
    # Variables that will hold the final vectors for similarity calculation
    user_vec_for_similarity = user_vec_raw.copy()
    dataset_vecs_for_similarity = dataset_vecs_raw.copy()

    if len(non_bool_numerical_cols) > 0:
        print(f"DEBUG: Scaling numerical columns: {non_bool_numerical_cols}") # For verification
        scaler = MinMaxScaler()

        scaled_dataset_numerical_part = scaler.fit_transform(dataset_vecs_raw[non_bool_numerical_cols])
        
        scaled_user_numerical_part = scaler.transform(user_vec_raw[non_bool_numerical_cols])
        
        dataset_vecs_for_similarity[non_bool_numerical_cols] = scaled_dataset_numerical_part
        user_vec_for_similarity[non_bool_numerical_cols] = scaled_user_numerical_part
    else:
        print("DEBUG: No numerical columns (excluding binary/OHE) found for scaling.")
    
    try:
        # Ensuring the vectors are 2D for cosine_similarity
        if user_vec_for_similarity.ndim == 1:
            user_vec_for_similarity_2d = user_vec_for_similarity.to_numpy().reshape(1, -1)
        else:
            user_vec_for_similarity_2d = user_vec_for_similarity.to_numpy()

        if dataset_vecs_for_similarity.ndim == 1: 
            dataset_vecs_for_similarity_2d = dataset_vecs_for_similarity.to_numpy().reshape(1, -1)
        else:
            dataset_vecs_for_similarity_2d = dataset_vecs_for_similarity.to_numpy()

        similarities = cosine_similarity(user_vec_for_similarity_2d, dataset_vecs_for_similarity_2d)
        
        similar_indices_sorted = np.argsort(similarities[0])[::-1]
        top_n_indices_from_all_students = all_students_features_processed.iloc[similar_indices_sorted[:top_n]].index
        
        return top_n_indices_from_all_students
        
    except Exception as e:
        print(f"Error calculating similarity: {e}")
        import traceback
        traceback.print_exc()
        return pd.Index([])

def generate_recommendations(user_input_dict, user_processed_features, user_prediction_proba,
                             all_students_features_processed, # PREPROCESSED features (no target)
                             all_students_original_values):   # ORIGINAL values + TARGET ('Depression')
    """
    Generates personalized recommendations.
    Args:
        user_input_dict (dict): Raw user input.
        user_processed_features (pd.DataFrame): PREPROCESSED features of the current user.
        user_prediction_proba (float): Predicted probability of depression for the user.
        all_students_features_processed (pd.DataFrame): DataFrame of ALL students' PREPROCESSED features.
        all_students_original_values (pd.DataFrame): DataFrame of ALL students' ORIGINAL values, including 'Depression' target.
    Returns:
        list: A list of recommendation dictionaries.
    """
    recommendations = []
    user_risk_level = get_risk_level(user_prediction_proba)

    # Direct Recommendations based on user input 
    if user_input_dict.get('Academic Pressure', 0) >= RECOMMENDATION_CATEGORIES["academic_pressure"]["high_threshold"]:
        recommendations.append({
            "category": "High Academic Pressure",
            "suggestions": np.random.choice(RECOMMENDATION_CATEGORIES["academic_pressure"]["suggestions"], 2, replace=False).tolist()
        })
    user_sleep_ordinal = sleep_map.get(user_input_dict.get('Sleep Duration'), DEFAULT_SLEEP_ORDINAL)
    if user_sleep_ordinal <= RECOMMENDATION_CATEGORIES["sleep_patterns"]["low_threshold"]:
        recommendations.append({
            "category": "Poor Sleep Patterns",
            "suggestions": np.random.choice(RECOMMENDATION_CATEGORIES["sleep_patterns"]["suggestions"], 2, replace=False).tolist()
        })
    if user_input_dict.get('Financial Stress', 0) >= RECOMMENDATION_CATEGORIES["financial_stress"]["high_threshold"]:
        recommendations.append({
            "category": "High Financial Stress",
            "suggestions": np.random.choice(RECOMMENDATION_CATEGORIES["financial_stress"]["suggestions"], 2, replace=False).tolist()
        })
    if user_input_dict.get('Study Satisfaction', 5) <= RECOMMENDATION_CATEGORIES["study_satisfaction"]["low_threshold"]: # Default high if not provided
        recommendations.append({
            "category": "Low Study Satisfaction",
            "suggestions": np.random.choice(RECOMMENDATION_CATEGORIES["study_satisfaction"]["suggestions"], 2, replace=False).tolist()
        })
    ws_hours = user_input_dict.get('Work/Study Hours', 0)
    if ws_hours >= RECOMMENDATION_CATEGORIES["work_study_hours"]["high_threshold_very_high"]:
        recommendations.append({
            "category": "Very High Work/Study Hours",
            "suggestions": np.random.choice(RECOMMENDATION_CATEGORIES["work_study_hours"]["suggestions_very_high"], 1, replace=False).tolist()
        })
    elif ws_hours >= RECOMMENDATION_CATEGORIES["work_study_hours"]["high_threshold_moderate"]:
         recommendations.append({
            "category": "High Work/Study Hours",
            "suggestions": np.random.choice(RECOMMENDATION_CATEGORIES["work_study_hours"]["suggestions_moderate"], 1, replace=False).tolist()
        })

    # Comparative Recommendations
    if user_risk_level in ["Medium", "High"] and not all_students_features_processed.empty:
        
        similar_student_indices = find_similar_students(
            user_processed_features,
            all_students_features_processed,
            top_n=30 
        )

        if len(similar_student_indices) > 0:
            similar_students_data_orig = all_students_original_values.loc[similar_student_indices]
            
            low_risk_similar_students_orig = similar_students_data_orig[similar_students_data_orig['Depression'] == 0]

            if not low_risk_similar_students_orig.empty:
                comparative_suggestions = []
                for field, props in COMPARISON_FIELDS.items():
                    user_val_str = user_input_dict.get(props['display_name']) 
                    user_val_numeric = None

                    if field == 'Sleep_Ordinal':
                        user_val_numeric = sleep_map.get(user_val_str, DEFAULT_SLEEP_ORDINAL)
                    elif user_val_str is not None : 
                        try:
                            user_val_numeric = float(user_val_str)
                        except ValueError:
                            continue 

                    if user_val_numeric is not None and field in low_risk_similar_students_orig.columns:
                        avg_low_risk_val = low_risk_similar_students_orig[field].median() 

                        significant_difference = False
                        comparison_text = ""

                        if props['lower_is_better']:
                            if user_val_numeric > avg_low_risk_val * 1.1:
                                significant_difference = True
                                comparison_text = f"lower their {props['display_name']} (average for them: {avg_low_risk_val:.1f})"
                        else: 
                            if user_val_numeric < avg_low_risk_val * 0.9: 
                                significant_difference = True
                                if field == 'Sleep_Ordinal':
                                    user_display_val = sleep_map_inv.get(int(user_val_numeric), "current level")
                                    avg_low_risk_display_val = sleep_map_inv.get(int(round(avg_low_risk_val)), "a healthy amount")
                                    comparison_text = f"achieve more '{avg_low_risk_display_val}' of sleep (you reported '{user_display_val}')"
                                else:
                                    comparison_text = f"report higher {props['display_name']} (average for them: {avg_low_risk_val:.1f})"
                        
                        if significant_difference:
                            comparative_suggestions.append(
                                f"Students similar to you who are doing well often manage to{comparison_text}. "
                                f"Consider exploring strategies in this area."
                            )
                
                if comparative_suggestions:
                     recommendations.append({
                         "category": "Insights from Similar Low-Risk Students",
                         "suggestions": comparative_suggestions[:3] # Show top 3 comparative insights
                     })

    # General Well-being Advice if Medium/High Risk 
    if user_risk_level in ["Medium", "High"] or not recommendations:
        general_wellbeing_recs = []
        if not any(r['category'] == "General Well-being" for r in recommendations): 
            general_wellbeing_recs.append("Consider talking to a campus counselor or a mental health professional.")
            
        potential_general_recs_keys = ["mindfulness_stress_reduction", "physical_activity", "social_connection"]
        existing_categories = [r['category'] for r in recommendations]
        
        added_general_count = 0
        for key in np.random.permutation(potential_general_recs_keys): # Shuffle to get variety
            if added_general_count < 2 and key.replace("_", " ").title() not in existing_categories:
                general_wellbeing_recs.append(np.random.choice(RECOMMENDATION_CATEGORIES[key]["suggestions"]))
                added_general_count +=1
        
        if general_wellbeing_recs:
             recommendations.append({
                "category": "General Well-being & Support",
                "suggestions": general_wellbeing_recs
            })
            
    return recommendations

# Main function for testing 
if __name__ == '__main__':
    
    sample_user_input = {
        'Age': 22, 'Gender': 'Female', 'City': 'Pune',
        'Academic Pressure': 5, 'CGPA': 7.5, 'Study Satisfaction': 2,
        'Sleep Duration': 'Less than 5 hours', 'Dietary Habits': 'Unhealthy',
        'Degree': 'B.Tech', 'Suicidal Thoughts': 'Yes', 'Work/Study Hours': 12,
        'Financial Stress': 4, 'Family History': 'No'
    }
    sample_user_proba = 0.75 
    mock_user_processed_features = pd.DataFrame(np.random.rand(1, 90), columns=[f'feature_{i}' for i in range(90)]) # Adjust 90 to your actual feature count
    mock_all_students_features_processed = pd.DataFrame(np.random.rand(100, 90), columns=[f'feature_{i}' for i in range(90)])

    mock_all_students_original_values = pd.DataFrame({
        'Age': np.random.randint(18, 35, 100),
        'Academic Pressure': np.random.randint(1, 6, 100),
        'Sleep_Ordinal': np.random.randint(0, 4, 100),
        'Financial Stress': np.random.randint(1, 6, 100),
        'Study Satisfaction': np.random.randint(1, 6, 100),
        'Work/Study Hours': np.random.randint(4, 15, 100),
        'Depression': np.random.randint(0, 2, 100) # 0 or 1
    })
    mock_all_students_features_processed.index = mock_all_students_original_values.index


    print("\nTesting Recommendation Generation\n")
    recommendations = generate_recommendations(
        sample_user_input,
        mock_user_processed_features,
        sample_user_proba,
        mock_all_students_features_processed,
        mock_all_students_original_values
    )

    print("\n--- Generated Recommendations (Test) ---")
    if recommendations:
        for rec_item in recommendations:
            print(f"\nCategory: {rec_item['category']}")
            for suggestion in rec_item['suggestions']:
                print(f"- {suggestion}")
    else:
        print("No recommendations generated.")