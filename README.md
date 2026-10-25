# Credit-Wise-Loan-Approval-Prediction-System
A supervised machine learning project that predicts whether a loan application will be approved or rejected from applicant data such as income, credit score, debt-to-income ratio, savings and employment details. It covers the full pipeline: data cleaning, EDA, encoding, scaling, model training and evaluation.

Data flow overview

Dataset
File: loan_approval_data.csv (1,000 applicants, 20 columns)
Target: Loan_Approved (Yes / No), about 70.2% No and 29.8% Yes, so the classes are imbalanced
Numeric features: Applicant_Income, Coapplicant_Income, Age, Dependents, Credit_Score, Existing_Loans, DTI_Ratio, Savings, Collateral_Value, Loan_Amount, Loan_Term
Categorical features: Employment_Status, Marital_Status, Loan_Purpose, Property_Area, Education_Level, Gender, Employer_Category
The raw data contains missing values (50 per column), which are imputed in the notebook.

Pipeline
Missing values: mean imputation for numeric columns, most-frequent imputation for categorical columns
EDA: class balance, category counts, income distributions, box plots for outliers, credit score and income against approval
Cleaning: drop Applicant_ID
Encoding: LabelEncoder for Education_Level and the target, OneHotEncoder(drop="first") for the other categorical columns
Correlation analysis: heatmap and correlation of every feature with Loan_Approved
Split and scale: 80/20 train-test split (random_state=42), then StandardScaler
Models: Logistic Regression, K-Nearest Neighbors (k=5) and Gaussian Naive Bayes
Feature engineering: adds DTI_Ratio_sq and Credit_Score_sq, replacing the original columns, then retrains the models

Key insights: correlation with loan approval

Correlation insights

Feature	Correlation with approval
Credit Score	+0.490 (strongest positive)
DTI Ratio	−0.472 (strongest negative)
Loan Amount	−0.130 (weak negative)

Applicants with a higher credit score and a lower debt-to-income ratio are much more likely to be approved.

Model evaluation

Model evaluation

Results on the held-out test set (200 applicants: 139 rejected, 61 approved):

Model	Accuracy	Precision	Recall	F1	Confusion matrix
Logistic Regression	100%	100%	100%	100%	[[139, 0], [0, 61]]
Naive Bayes	100%	100%	100%	100%	[[139, 0], [0, 61]]
KNN (k=5)	96.0%	98.2%	88.5%	93.1%	[[138, 1], [7, 54]]

Logistic Regression and Naive Bayes tie for the best result. KNN is slightly behind and misses 7 approved applicants.

Limitations
Perfect scores need scrutiny. 100% on all metrics is unusual for real lending data. It can point to data leakage or to a synthetic dataset where approval follows a fixed rule. Treat these results as a demonstration of the pipeline, not as proof of real-world performance.
The test set is small (200 rows), so metrics can shift noticeably with a different split.
Cross-validation and hyperparameter tuning are not included yet.
Imputing the target column is not ideal. Rows with a missing Loan_Approved are better dropped.

Tech stack

Python, Pandas, NumPy, Scikit-learn, Seaborn, Matplotlib, Jupyter Notebook

Future work
Check for data leakage and re-evaluate
Add cross-validation and hyperparameter tuning
Try tree-based models (Random Forest, XGBoost) and handle the class imbalance explicitly (class weights or SMOTE)
Build a small web app for live predictions
