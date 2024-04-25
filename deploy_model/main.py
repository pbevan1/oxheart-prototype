from flask import Flask, request, jsonify, abort
from google.cloud import storage
import joblib
import os
from io import BytesIO
from google.cloud import aiplatform
from urllib.parse import urlparse

from arguments import parse_args

app = Flask(__name__)
args = parse_args()
PROJECT_ID = "pb-sandbox-1"


def get_newest_model_url(project_id, location, model_display_name):
    """Load the latest model version from Vertex AI Model Registry."""
    aiplatform.init(project=project_id, location=location)

    # Fetch models by display name
    filter_expression = f'display_name="{model_display_name}"'
    models = aiplatform.Model.list(
        filter=filter_expression, order_by="create_time desc"
    )

    if models:
        latest_model = models[0]  # Assumes the first model is the latest
        model_artifact_uri = latest_model.gca_resource.artifact_uri
        parsed_uri = urlparse(model_artifact_uri)
        bucket_name = parsed_uri.netloc
        model_path = parsed_uri.path.lstrip("/") + "/model.joblib".lstrip(
            parsed_uri.netloc
        )

        # Assuming the model is stored as a joblib file in the artifact URI
        return model_path, bucket_name
    else:
        raise Exception("No models found with the display name provided.")


def load_model_from_gcs(bucket_name, blob_name):
    """Load a joblib model file from a Google Cloud Storage bucket."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    buffer = blob.download_as_bytes()
    buffer_io = BytesIO(buffer)
    return joblib.load(buffer_io)


def require_api_key(f):
    """Decorator to enforce API key authentication."""

    def decorated(*args, **kwargs):
        api_key = request.headers.get("X-API-KEY")
        if api_key and api_key == os.getenv("API_KEY"):
            return f(*args, **kwargs)
        else:
            abort(401, "Unauthorized, please provide a valid API key.")

    return decorated


model_url, bucket_name = get_newest_model_url(
    PROJECT_ID, args.LOCATION, args.PIPELINE_NAME
)
print(f"Model URL: {model_url}")
print(f"Bucket Name: {bucket_name}")
model = load_model_from_gcs(bucket_name, model_url)


@app.route("/predict/", methods=["GET", "POST"])
@require_api_key
def predict():
    if request.method == "POST":
        data = request.get_json()  # Get JSON data from POST request
        features_str = data.get("features", "")
    else:
        features_str = request.args.get(
            "features", ""
        )  # Get data from query parameters for GET requests

    if not model:
        abort(503, "Model not loaded")

    # Split the features string into a list and convert to float
    try:
        features = [float(f) for f in features_str.split(",")]
    except ValueError:
        abort(400, "Features must be a comma-separated list of floats.")

    # Check if the number of features is exactly 12
    if len(features) != 12:
        abort(
            400,
            """Exactly 12 features are required: ["age","sex","chest_pain_type","resting_blood_pressure",
              "chol","fasting_blood_sugar","resting_ECG","max_heart_rate","exang","slope","number_vessels_flourosopy","thal"]""",
        )

    try:
        prediction = model.predict([features])
        return jsonify({"prediction": int(prediction[0])})
    except Exception as e:
        abort(400, str(e))


if __name__ == "__main__":
    app.run()
