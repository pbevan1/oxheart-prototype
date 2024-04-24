import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--LOCATION",
        type=str,
        help="GCP location to run pipeline in.",
        default="europe-west1",
    )
    parser.add_argument(
        "--MAIN_BUCKET_NAME",
        type=str,
        help="Name of the main GCP storage bucket to work in.",
        default="oxheart",
    )
    parser.add_argument(
        "--PIPELINE_NAME",
        type=str,
        help="What to call the pipeline.",
        default="mf-oxheart-prototype",
    )
    parser.add_argument(
        "--PIPELINE_ROOT",
        type=str,
        help="Where to store the pipeline in gcs.",
        default="gs://oxheart/heart/pipelines/",
    )
    parser.add_argument(
        "--DATASET_ID",
        type=str,
        help="What dataset the training data is in.",
        default="oxheart_prototype",
    )
    parser.add_argument(
        "--TABLE_ID",
        type=str,
        help="What table the training data is in.",
        default="heart_temp",
    )
    parser.add_argument(
        "--COL_TRAINING",
        nargs="+",
        help="What columns are the training data.",
        default=[
            "age",
            "sex",
            "chest_pain_type",
            "resting_blood_pressure",
            "chol",
            "fasting_blood_sugar",
            "resting_ECG",
            "max_heart_rate",
            "exang",
            "slope",
            "number_vessels_flourosopy",
            "thal",
        ],
    )
    parser.add_argument(
        "--COL_LABEL",
        type=str,
        help="What column is the ground truth.",
        default="target",
    )
    parser.add_argument(
        "--ENABLE_CACHING",
        action="store_true",
        help="Whether to enable caching for Vertex AI pipeline components.",
        default=False,
    )

    args, _ = parser.parse_known_args()
    return args
