import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
import argparse
import logging

# Setup logging (biar log-nya rapi di CI/CD)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train(data_path, n_estimators, max_depth):
    # Tracking via local file (penting untuk GitHub Action/docker)
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("CI-Pipeline")

    # Enable autolog (otomatis log params, metrics, artifacts standar)
    mlflow.sklearn.autolog(log_input_examples=True, log_model_signatures=True)

    # Load dan siapkan data
    df = pd.read_csv(data_path)
    X = df.drop("Personality", axis=1)
    X = X.astype({col: 'float64' for col in X.select_dtypes('int').columns})  
    y = df["Personality"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    with mlflow.start_run():
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        )
        model.fit(X_train, y_train)

        # Prediksi dan evaluasi
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        mlflow.log_metric("test_accuracy", acc)

        # Logging eksplisit untuk signature dan contoh input
        signature = infer_signature(X_test, y_pred)
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            input_example=X_test.iloc[:3],
            registered_model_name="PersonalityClassifier"
        )

        logger.info(f" Model trained & logged with accuracy: {acc:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="personality.csv")
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=10)
    args = parser.parse_args()

    train(args.data_path, args.n_estimators, args.max_depth)
