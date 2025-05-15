# Student Depression Risk Prediction System (C++ Implementation)

## Project Overview

This project is a C++ application designed to predict the likelihood of depression in students based on various academic, lifestyle, and stress-related factors. It utilizes a Random Forest classifier implemented from scratch in C++ and provides personalized recommendations through a Command Line Interface (CLI). The system also maintains a simple, local history of predictions for the user.

The primary goal is to offer students a tool for self-awareness regarding their mental well-being and to provide actionable, albeit general, advice. **This system is not a diagnostic tool and is not a substitute for professional medical advice.**

## Features

- **Risk Prediction:** Predicts depression risk (High/Low) based on user input.
- **Probability Score:** Provides an estimated probability for the high-risk category.
- **Personalized Recommendations:** Offers suggestions based on predicted risk and specific input factors (e.g., academic pressure, sleep).
- **Single-User History:** Saves prediction history (timestamp, probability, risk level) to a local text file (`risk_history.txt`) and displays recent entries.
- **Command Line Interface (CLI):** All interactions are through a text-based console interface.
- **From-Scratch Random Forest:** The core machine learning model (Decision Trees and Random Forest ensemble) is implemented in C++ without reliance on external ML libraries for the algorithm itself.

## Technology Stack

- **Programming Language:** C++ (compiled with a C++17 standard-compliant compiler like g++)
- **Core Libraries:** Standard C++ Library (for I/O, data structures, algorithms, random number generation, time).
- **Data Source:** A preprocessed CSV file (`cleaned_student_data.csv`) containing numerical features (including one-hot encoded categoricals from an initial Python preprocessing step).

## How it Works (High-Level)

1.  **Data Loading:** The application loads the `cleaned_student_data.csv`.
2.  **Data Splitting:** The loaded data is split into training and testing sets.
3.  **Model Training:** A Random Forest model (an ensemble of custom-built Decision Trees) is trained on the training data. The Decision Trees use Gini impurity for splitting and support feature subsampling. The Random Forest uses bootstrapping.
4.  **Model Evaluation:** The trained model is evaluated on the test set, displaying accuracy and a confusion matrix.
5.  **Interactive Prediction:**
    - The user is prompted to enter simplified data (e.g., age, academic pressure, sleep habits).
    - The C++ application transforms these simplified inputs into the full feature vector expected by the trained model (handling derived features and one-hot encoding logic internally for CLI input).
    - The Random Forest model predicts the risk class and probability.
    - The current prediction is saved to `risk_history.txt`.
    - Past history is displayed to the user.
    - Recommendations are generated based on the prediction and specific inputs.
    - Results are displayed in the CLI.

## Setup and Compilation

**Prerequisites:**

- A C++ compiler supporting C++17 (e.g., g++, Clang).
- The dataset file `cleaned_student_data.csv` (must be in the same directory as the executable or path adjusted in code).

**Compilation (using g++):**

```shell
g++ -std=c++17 -Wall -Wextra -O2 -o depression_classification main.cpp
```

- std=c++17: Specifies the C++ standard.
- Wall -Wextra: Enable most compiler warnings (recommended).
- O2: Optimization level.
- o depression_classification: Names the output executable.
- main.cpp: Your main source file.

## How to Run

1.  **Compile the code** as shown in the "Setup and Compilation" section above.
    ```bash
    g++ -std=c++17 -Wall -Wextra -O2 -o depression_classification main.cpp
    ```
2.  Ensure the `cleaned_student_data.csv` file is in the **same directory** as your compiled executable (e.g., `depression_classification`).
3.  **Run the executable** from your terminal:
    ```bash
    ./depression_classification
    ```
4.  **Follow the on-screen prompts** in the CLI to:
    - Observe model training and initial evaluation results.
    - Choose to make new predictions when prompted.
    - View your past prediction history (if any exists).
    - Enter your data for the simplified list of features.
    - View your predicted risk level, the associated probability, and personalized recommendations.

## How to Run Unit Test

1.  **Compile the unit test code**
    ```bash
    g++ -std=c++11 unit_test.cpp -o test
    ```
2.  **Run the executable** from your terminal:
    ```bash
    ./test
    ```

## Input Data for CLI

The Command Line Interface will prompt you for the following simplified inputs:

- **Age:** (e.g., `22`)
- **Academic Pressure:** (Range: 1-5, e.g., `4`)
- **CGPA:** (Range: 0.0-10.0, e.g., `7.5`)
- **Study Satisfaction:** (Range: 1-5, e.g., `3`)
- **Suicidal Thoughts:** (0 for No / 1 for Yes, e.g., `0`)
- **Work/Study Hours per day:** (e.g., `8`)
- **Financial Stress:** (Range: 1-5, e.g., `2`)
- **Family History of Mental Illness:** (0 for No / 1 for Yes, e.g., `1`)
- **Sleep Ordinal:**
  - `0` for "Less than 5 hours"
  - `1` for "5-6 hours"
  - `2` for "7-8 hours"
  - `3` for "More than 8 hours"
    (e.g., `2`)
- **Gender:** (0 for Female/Other, 1 for Male, e.g., `1`)
- **Degree Type:**
  - `0` for Science/Tech Related Field (e.g., Engineering, Medical, Pure Sciences)
  - `1` for Non-Science/Arts/Business/Law Field (e.g., Arts, Humanities, Business, Law, Management, Education)
    (e.g., `0`)

The program internally processes these simplified inputs and converts them into the full feature set that the trained Random Forest model expects.

## Future Work (Potential Phase II Enhancements)

- **Graphical User Interface (GUI):** Develop a more user-friendly GUI
- **Advanced Recommendation Engine:** Improve the recommendation system to provide more detailed, context-aware, and potentially evidence-backed advice.
- **User Accounts & Personalized History:** Implement user accounts to allow multiple users to securely track their individual risk history over time, requiring data storage and authentication.
- **Hyperparameter Optimization:** Integrate or develop methods for more systematic hyperparameter tuning of the C++ Random Forest model.
- **Robustness:** Enhance error handling, input validation, and logging throughout the application.
- **Model Iteration:** Explore further feature engineering within C++ or consider alternative C++ based modeling approaches if needed.

## Disclaimer

This tool is for educational and informational purposes only. It is **not a substitute for professional medical advice, diagnosis, or treatment.** If you are concerned about your mental health or are experiencing distress, please consult a qualified healthcare provider or mental health professional.

## 🤝 Contributing

**Contributors**

- Diwash Bhattarai
- Faiza Khan
- Durapatie (Priya) Ramdat

---
