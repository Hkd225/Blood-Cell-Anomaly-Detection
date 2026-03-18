# Blood Cell Anomaly & Disease Detection
## Advanced Machine Learning Pipeline with Leakage Prevention
### By Muhammad Auffa Hakim Aditya

This project presents a highly rigorous, production-ready Machine Learning pipeline designed to classify blood cell types into broader disease categories (e.g., Leukemia, Anemia, Infection). Utilizing a 2025 dataset from Kaggle, this project emphasizes robust MLOps practices, including automated data leakage prevention, patient-level group validation, and rigorous sanity checks to ensure model reliability in a medical context.

The project was developed by Muhammad Auffa Hakim Aditya to demonstrate advanced tabular data engineering, highlighting how to handle complex medical datasets where preventing spurious correlations and data leakage is just as important as achieving high accuracy.

------------------------------------------------------------

PROJECT OBJECTIVES

1. Automated Target Mapping: Dynamically map fine-grained blood cell types (e.g., Blast Cells, Elliptocytes) into primary disease categories (Leukemia, Anemia, Normal, etc.) using either a reference CSV or a predefined rule-based dictionary.
2. Strict Data Leakage Prevention: Automatically detect and drop explicit ID columns (patient_id, slide_id) and potential target-leaking columns (scores, probabilities, diagnosis labels) using keyword heuristics.
3. Group-Aware Validation: Implement `GroupShuffleSplit` and `GroupKFold` if group identifiers (like `patient_id` or `slide_id`) are present, ensuring cells from the same patient do not leak across the train/test split.
4. Multi-Model Cross-Validation: Train and evaluate Random Forest, Logistic Regression, SVM, and XGBoost using a centralized Scikit-Learn `Pipeline` and `ColumnTransformer`.
5. Sanity Check (Label Shuffling): Perform a rigorous baseline test by deliberately shuffling the training labels to ensure the model's accuracy drops to random guessing, proving the absence of hidden data leakage.
6. Unified Artifact Export: Bundle all models, preprocessors, encoders, metadata, and evaluation reports into a structured `.zip` archive.

------------------------------------------------------------

DATASET INFORMATION

Source          : Kaggle (alitaqishah/blood-cell-anomaly-detection-2025)
Domain          : Healthcare / Hematology
Target Variable : disease_level

Disease Mapping Rule:
- Normal: Neutrophil, Lymphocyte, Monocyte, Eosinophil, Basophil, Normal RBC, Platelet
- Leukemia: Blast Cell, Prolymphocyte
- Anemia: Elliptocyte, Schistocyte, Spherocyte, Target Cell
- Sickle Cell Disease: Sickle Cell
- Infection: Hypersegmented Neutrophil, Toxic Granulation, Reactive Lymphocyte
- Artefact: Smudge Cell, Artefact

------------------------------------------------------------

MACHINE LEARNING PIPELINE ARCHITECTURE

1. Dynamic Preprocessing (`ColumnTransformer`):
   - Numerical Features: `SimpleImputer(strategy="median")` + `MinMaxScaler()`
   - Categorical Features: `SimpleImputer(strategy="most_frequent")` + `OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)`

2. Model Ensembles (with balanced class weights):
   - Random Forest (200 estimators)
   - Logistic Regression (L2 penalty, LBFGS solver)
   - Support Vector Machine (RBF kernel, Probability enabled)
   - XGBoost (Softprob objective, depth 6)

3. Advanced Evaluation Metric:
   Due to the likely imbalance in medical datasets, the pipeline strictly evaluates models based on Macro F1-Score and Balanced Accuracy, rather than standard accuracy.

------------------------------------------------------------

MLOps & MODEL RELIABILITY

- Automated Leakage Filter: The code scans feature names for keywords like "confidence", "prediction", "anomaly_label" and dynamically drops them before training.
- Sanity Check Protocol: Before finalizing, the pipeline trains a baseline model on randomly shuffled targets. If the F1-Score remains unusually high (>0.30), the system triggers a warning that hidden shortcuts or spurious correlations exist in the data.

------------------------------------------------------------

DEPLOYMENT ARTIFACTS

The script automatically generates a highly structured folder and compresses it into a `.zip` archive containing:

blood-cell-anomaly-detection-2025_models/
    ├── model_random_forest.pkl
    ├── model_logistic_regression.pkl
    ├── model_svm.pkl
    ├── model_xgboost.pkl
    ├── best_model.pkl (The absolute best performing pipeline)
    ├── target_encoder.pkl
    ├── feature_columns.pkl / categorical_cols.pkl / numeric_cols.pkl
    ├── dropped_columns.pkl
    ├── training_metadata.json (Complete hyperparameters and evaluation scores)
    └── classification_report.txt

------------------------------------------------------------

INSTALLATION

Install the required dependencies:

pip install pandas numpy scikit-learn xgboost kagglehub joblib

------------------------------------------------------------

HOW TO RUN

1. Clone this repository.
2. Run the Python script. The script handles data fetching, dynamic target mapping, leakage dropping, cross-validation, sanity checking, and artifact zipping automatically.

------------------------------------------------------------

AUTHOR

Muhammad Auffa Hakim Aditya

This project was developed as an exploration of:
- Medical Machine Learning
- Advanced Scikit-Learn Pipelines
- Data Leakage Prevention & Sanity Checking
- GroupKFold Cross Validation
- Automated MLOps Artifact Generation

------------------------------------------------------------

KEYWORDS 

- Muhammad Auffa Hakim Aditya
- Blood Cell Classification
- Anomaly Detection
- Scikit-Learn Pipeline
- XGBoost
- MLOps Data Leakage
- GroupKFold
- Machine Learning Portfolio
