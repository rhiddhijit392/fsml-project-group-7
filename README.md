Hello , this is our Full Stack Machine Learning Project. In this we are going to use the supervised machine learning techniques . The techniques are XGboost, Random Forest, Logistic Regression.


Changes, assumptions, and suggestions from DE:
1. data_loader.py: assumes data is stored in data/(dataset_name).csv
2. preprocessor.py: rows grouped by company name and then ordered by date, NaN replaced with last valid value
3. features: added 5 Engineered features, then dropped NaN rows resulted from grouping
5. .gitignore: won't accidently push dataset.csv files & python or vscode files
6. Remember to update "requirements.txt" & "pipeline.py" after your part of work is done
7. Incase anyone runs in python version mismatch errors: I'm currently using python 3.12.13 (beacuse tensorflow doesn't support python 3.14 yet)

Note for MLE: 'Next_Day_UP' feature might be useful for classification
