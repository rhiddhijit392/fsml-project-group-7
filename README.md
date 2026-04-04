Hello , this is our Full Stack Machine Learning Project. In this we are going to use the supervised machine learning techniques . The techniques are XGboost, Random Forest, Logistic Regression.


Changes, assumptions, and suggestions from DE:
1. data_loader.py: assumes data is stored in data/(dataset_name).csv
2. preprocessor.py: rows grouped by company name and then ordered by date, NaN replaced with last valid value
3. features: added 5 Engineered features, then dropped NaN rows resulted from grouping
5. .gitignore: won't accidently push dataset.csv files & python or vscode files
6. Remember to update "requirements.txt" & "pipeline.py" after your part of work is done
7. Incase anyone runs in python version mismatch errors: I'm currently using python 3.12.13 (beacuse tensorflow doesn't support python 3.14 yet)

Note for MLE: 'Next_Day_UP' feature might be useful for classification
ML Role Now:- 
 **1) src/utils.py**=>Shared utility library for the ML pipeline. Handles logging setup, config loading, model save/load via joblib, regression metrics computation (MAE, RMSE, R²), model comparison, and directory management.


**2) src/train.py**=>Trains two regression models — Random Forest and XGBoost — to predict the next day's closing stock price. Runs the full data pipeline, removes data leakage, splits data chronologically (80/20), compares both models, and saves the best model to models/model_v1.pkl. 
**3) src/evaluate.py**=>Loads the saved best model and evaluates it on the held-out test set. Reports MAE, RMSE, and R² score with a human-readable verdict. Generates two plots — Actual vs Predicted scatter and Residuals Distribution — saved to outputs/evaluation_plots.png. 
**4)src/predict.py**=>Loads the saved model and generates next day closing price predictions for the most recent trading data per company. Outputs a formatted table with predicted price, expected dollar change, and UP/DOWN direction signal. Saves full results to outputs/predictions.csv 
## Model
Model is not tracked in git due to file size.
Run python src/train.py to regenerate model_v1.pkl locally.
