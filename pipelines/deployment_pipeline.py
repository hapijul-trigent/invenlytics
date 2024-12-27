import mlflow
from mlflow.tracking import MlflowClient
import subprocess

def deploy_disruption_forecaster(uri: str, model_name: str, stage: str = "Production", port: int = 5000):
    """
    Deploys an MLflow model from a given runs URI.

    Parameters:
        uri (str): The MLflow model URI (e.g., `runs:/<run-id>/model`).
        model_name (str): The name of the model to register or retrieve in the Model Registry.
        stage (str): The stage to transition the model to (default: "Production").
        port (int): The port to serve the model (default: 5000).
    """
    try:
        client = MlflowClient()
        print(f"Registering model: {model_name} from URI: {uri}")
        try:
            client.get_registered_model(model_name)
        except mlflow.exceptions.RestException:
            print(f"Model {model_name} does not exist. Creating a new registered model.")
            client.create_registered_model(model_name)

        model_version = client.create_model_version(name=model_name, source=uri, run_id=uri.split("/")[1])
        print(f"Model version {model_version.version} created successfully.")
        print(f"Transitioning model {model_name} to stage: {stage}")
        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage=stage,
        )
        print(f"Model {model_name} transitioned to {stage} stage.")

        
        def serve_model():
            print(f"Serving model {model_name} at stage: {stage} on port {port}")
            model_serving_command = [
                "mlflow", "models", "serve",
                "-m", f"models:/{model_name}/{stage}",
                "--port", str(port)
            ]
            subprocess.run(model_serving_command)

        serve_model()

    except Exception as e:
        print(f"An error occurred during deployment: {str(e)}")
