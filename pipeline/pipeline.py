import sys
import os

# Add project root to Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import load_data
from src.preprocess import preprocess
from src.features import build_features
from src.train import run_training
from src.evaluate import run_evaluation
from src.predict import run_prediction

def run_pipeline():
    # Step 1: Load data
    df = load_data()
    print("Columns loaded from data:", df.columns.tolist())

    # Step 2: Preprocess
    df = preprocess(df)
    print("Columns after preprocess:", df.columns.tolist()) 

    # Step 3: Feature engineering
    df = build_features(df)
    print("Columns after build_features:", df.columns.tolist())
    
    #Pipeline upto DE work
    print(f"Data Pipeline complete!\n processed data: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Sample Data Stream:\n{df.head()}\n{'='*55}\n")

    # Step 4: Train model
    model, X_train, X_test, y_train, y_test, test_idx = run_training(df)
    print("Training pipeline complete!\n", "="*55)

    #step 5: Evaluation
    run_evaluation(df, model=model, X_test=X_test, y_test=y_test, test_indices=test_idx)
    print("Evaluation complete!\n", "="*55)

    #step 6: Prediction
    run_prediction(df, model=model)
    print("prediction complete!\n", "="*55)
    
    return df

if __name__ == "__main__":
    df = run_pipeline()