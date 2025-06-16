import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split  
import mlflow
import argparse
import os
from sklearn.metrics import accuracy_score, classification_report

def train(data_path, n_estimators, max_depth):
    # Selalu gunakan filesystem untuk tracking dalam CI/CD
    mlflow.set_tracking_uri("file:///mlruns")
    mlflow.set_experiment("Personality-Classification")
    
    # Autolog dengan opsi tambahan
    mlflow.sklearn.autolog(
        log_input_examples=True,
        log_model_signatures=True,
        log_models=False  
    )

    # Load data
    df = pd.read_csv(data_path)
    X = df.drop("Personality", axis=1)
    y = df["Personality"]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    with mlflow.start_run():
        # Log parameters
        mlflow.log_params({
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "data_path": data_path,
            "train_size": len(X_train),
            "test_size": len(X_test)
        })
        
        # Train model
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        )
        model.fit(X_train, y_train)

        # Evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        
        # Log classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        for label, metrics in report.items():
            if isinstance(metrics, dict):
                for metric, value in metrics.items():
                    mlflow.log_metric(f"{label}_{metric}", value)

        # Log model secara manual
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name="PersonalityClassifier",
            input_example=X_train[:5],
            signature=mlflow.models.infer_signature(X_train, y_pred)
        )
        
        print(f"Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=10)
    parser.add_argument("--data_path", type=str, default="personality.csv")
    args = parser.parse_args()
    train(args.data_path, args.n_estimators, args.max_depth)
