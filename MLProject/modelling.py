import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split  
import mlflow
import argparse
from sklearn.metrics import accuracy_score, classification_report

def train(data_path, n_estimators, max_depth):
    # Setup MLflow
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("CI-Pipeline")
    
    # Enable autolog with additional options
    mlflow.sklearn.autolog(
        log_input_examples=True,
        log_model_signatures=True,
        log_models=True
    )

    # Load data
    df = pd.read_csv(data_path)
    X = df.drop("Personality", axis=1)
    y = df["Personality"]

    # Split data (80:20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42
    )

    with mlflow.start_run():
        # Log data parameters
        mlflow.log_param("data_path", data_path)
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size", len(X_test))
        
        # Initialize and train model
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        )
        model.fit(X_train, y_train)

        # Evaluate model
        y_pred = model.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_pred)
        
        # Log metrics
        mlflow.log_metric("test_accuracy", test_accuracy)
        
        # Generate and log classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        for key in report:
            if isinstance(report[key], dict):
                for metric in report[key]:
                    mlflow.log_metric(f"{key}_{metric}", report[key][metric])
            else:
                mlflow.log_metric(key, report[key])
        
        # Log model explicitly
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name="PersonalityClassifier"
        )
        
        print(f"Test Accuracy: {test_accuracy:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=10)
    parser.add_argument("--data_path", type=str, default="personality.csv")
    args = parser.parse_args()
    train(args.data_path, args.n_estimators, args.max_depth)