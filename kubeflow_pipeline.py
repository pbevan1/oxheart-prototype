import json
from typing import NamedTuple
from datetime import datetime
import google.cloud.aiplatform as aiplatform
from kfp import dsl, compiler
from kfp.dsl import (component, Input, Model, Output, Dataset, 
                        Artifact, OutputPath, ClassificationMetrics, 
                        Metrics, InputPath)


GCP_BIGQUERY = "google-cloud-bigquery==3.20.1"
PANDAS = "pandas==2.0.3"
SKLEARN = "scikit-learn==1.4.2"
NUMPY = "numpy==1.26.4"
BASE_IMAGE = "us-docker.pkg.dev/deeplearning-platform-release/gcr.io/sklearn-cpu"

TIMESTAMP = datetime.now().strftime("%Y%m%d%H%M%S")
PROJECT_ID = "pb-sandbox-1"
LOCATION = "europe-west1"
PIPELINE_ROOT = "gs://oxheart/heart/"
SERVICE_ACCOUNT = "pb-testing@pb-sandbox-1.iam.gserviceaccount.com"  # Move this to a credentials file and read it
PIPELINE_NAME = "mf-oxheart-prototype"
JOBID = f"ml-pipeline-{TIMESTAMP}"
ENABLE_CACHING = False
TEMPLATE_PATH = f"{PIPELINE_NAME}.json"

@component(base_image=BASE_IMAGE, packages_to_install=[GCP_BIGQUERY])
def query_to_table(
    query: str,
    project_id: str,
    dataset_id: str,
    table_id: str,
    location: str = "EU",
    query_job_config: dict = None,
) -> None:
    """
    Run the query and create a new BigQuery table
    """
    
    import google.cloud.bigquery as bq
    
    # Configure your query job
    job_config = bq.QueryJobConfig(destination=f"{project_id}.{dataset_id}.{table_id}", 
                                   **query_job_config)
    
    # Initiate the Big Query client to connect with the project
    bq_client = bq.Client(project=project_id, 
                          location=location)
    
    # Generate the query with all the job configurations
    query_job = bq_client.query(query, job_config=job_config)
    query_job.result()

    print(f"Query job with ID {query_job.job_id} finished.")


@component(base_image=BASE_IMAGE, packages_to_install=[GCP_BIGQUERY])
def extract_table_to_gcs(
    project_id: str,
    dataset_id: str,
    table_id: str,
    dataset: Output[Dataset],
    location: str = "EU",
) -> None:
    """
    Extract a Big Query table into Google Cloud Storage.
    """

    import logging
    import os
    import google.cloud.bigquery as bq

    # Get the table generated on the previous component
    full_table_id = f"{project_id}.{dataset_id}.{table_id}"
    table = bq.table.Table(table_ref=full_table_id)

    # Initiate the Big Query client to connect with the project
    job_config = bq.job.ExtractJobConfig(**{})
    client = bq.client.Client(project=project_id, location=location)

    # Submit the extract table job to store on GCS
    extract_job = client.extract_table(table, dataset.uri)


@component(
    base_image=BASE_IMAGE, packages_to_install=[PANDAS, SKLEARN]
)
def create_sets(
    data_input: Input[Dataset],
    dataset_train: OutputPath(),
    dataset_test: OutputPath(),
    col_label: str,
    col_training: list
    ) -> NamedTuple("Outputs", [("dict_keys", dict), ("shape_train", int), ("shape_test", int)]):
    
    
    """
    Split data into train and test sets.
    """

    import logging
    import pickle

    import pandas as pd
    from sklearn import model_selection

    def convert_labels_to_categories(labels):
        """
        The function returns a dictionary with the encoding of labels.
        :returns: A Pandas DataFrame with all the metrics
        """
        try:
            dic_keys = {k: label for k, label in enumerate(sorted(labels.unique()))}
            dic_vals = {label: k for k, label in enumerate(sorted(labels.unique()))}
            return dic_vals, dic_keys
        except Exception as e:
            print(f'[ERROR] Something went wrong that is {e}')
        return {}, {}


    df_ = pd.read_csv(data_input.path)
    
    df_.dropna(inplace=True)

    logging.info(f"[START] CREATE SETS, starts with an initial shape of {df_.shape}")

    if len(df_) != 0:

        yy = df_[col_label]
        dic_vals, dic_keys = convert_labels_to_categories(yy)

        yy = yy.apply(lambda v: dic_vals[v])
        xx = df_[col_training]

        x_train, x_test, y_train, y_test = model_selection.train_test_split(xx, yy, test_size=0.2, random_state=0, stratify=yy)

        x_train_results = {'x_train': x_train, 'y_train': y_train}
        x_test_results = {'x_test': x_test, 'y_test': y_test}

        with open(dataset_train + f".pkl", 'wb') as file:
            pickle.dump(x_train_results, file)

        with open(dataset_test + ".pkl", 'wb') as file:
            pickle.dump(x_test_results, file)

        logging.info(f"[END] CREATE SETS, data set was split")

        return (dic_keys, len(x_train), len(x_test))

    else:
        logging.error(f"[END] CREATE SETS, data set is empty")
        return (None, None, None)


@component(
    base_image=BASE_IMAGE, packages_to_install=[SKLEARN, PANDAS]
)
def train_model(
    training_data: InputPath(),
    model: Output[Model],
) -> None:
    """
    Train a classification model.
    """
        
    import logging
    import os
    import pickle
    import joblib
    import numpy as np
    from sklearn.linear_model import LogisticRegression

    logging.getLogger().setLevel(logging.INFO)

    # you have to load the training data
    with open(training_data + ".pkl", 'rb') as file:
        train_data = pickle.load(file)

    X_train = train_data['x_train']
    y_train = train_data['y_train']
    
    logging.info(f"X_train shape {X_train.shape}")
    logging.info(f"y_train shape {y_train.shape}")

    logging.info("Starting Training...")
    
    clf = LogisticRegression(n_jobs=-1, random_state=42)
    train_model = clf.fit(X_train, y_train)

    # ensure to change GCS to local mount path
    os.makedirs(model.path, exist_ok=True)

    # ensure that you save the final model as a .joblib
    logging.info(f"Save model to: {model.path}")
    joblib.dump(train_model, model.path + "/model.joblib")


@component(
    base_image=BASE_IMAGE, packages_to_install=[PANDAS]
)
def predict_model(
    test_data: InputPath(),
    model: Input[Model],
    predictions: Output[Dataset],
) -> None:
    
    
    """
    Create the predictions of the model.
    """    

    import logging
    import os
    import pickle
    import joblib
    import pandas as pd

    logging.getLogger().setLevel(logging.INFO)

    # you have to load the test data
    with open(test_data + ".pkl", 'rb') as file:
        test_data = pickle.load(file)

    X_test = test_data['x_test']
    y_test = test_data['y_test']

    # load model
    model_path = os.path.join(model.path, "model.joblib")
    model = joblib.load(model_path)
    y_pred = model.predict(X_test)

    # predict and save to prediction column
    df = pd.DataFrame({
        'class_true': y_test.tolist(),
        'class_pred': y_pred.tolist()}
    )

    # save dataframe
    df.to_csv(predictions.path, sep=",", header=True, index=False)


@component(
    base_image=BASE_IMAGE, packages_to_install=[PANDAS, NUMPY]
)
def evaluation_metrics(
    predictions: Input[Dataset],
    metrics_names: list,
    dict_keys: dict,
    metrics: Output[ClassificationMetrics],
    kpi: Output[Metrics],
    eval_metrics: Output[Metrics]
) -> None:
    
    """
    Create the evaluation metrics.
    """ 
    import json
    import logging
    from importlib import import_module
    import numpy as np
    import pandas as pd

    results = pd.read_csv(predictions.path)
    
    # Encode the predictions model
    results['class_true_clean'] = results['class_true'].astype(str).map(dict_keys)
    results['class_pred_clean'] = results['class_pred'].astype(str).map(dict_keys)
    
    # To fetch metrics from sklearn
    module = import_module(f"sklearn.metrics")
    metrics_dict = {}
    for each_metric in metrics_names:
        metric_func = getattr(module, each_metric)
        if each_metric == 'f1_score':
            metric_val = metric_func(results['class_true'], results['class_pred'], average=None)
        else:
            metric_val = metric_func(results['class_true'], results['class_pred'])
        
        # Save metric name and value
        metric_val = np.round(np.average(metric_val), 4)
        metrics_dict[f"{each_metric}"] = metric_val
        kpi.log_metric(f"{each_metric}", metric_val)
        
        # dumping kpi metadata to generate the metrics kpi
        with open(kpi.path, "w") as f:
            json.dump(kpi.metadata, f)
        logging.info(f"{each_metric}: {metric_val:.3f}")

    # dumping metrics_dict to generate the metrics table
    with open(eval_metrics.path, "w") as f:
        json.dump(metrics_dict, f)

    # to generate the confusion matrix plot
    confusion_matrix_func = getattr(module, "confusion_matrix")
    metrics.log_confusion_matrix(list(dict_keys.values()),
        confusion_matrix_func(results['class_true_clean'], results['class_pred_clean']).tolist(),)
    
    # dumping metrics metadata
    with open(metrics.path, "w") as f:
        json.dump(metrics.metadata, f)


@dsl.pipeline(name=PIPELINE_NAME, pipeline_root=PIPELINE_ROOT)
def oxheart_prototype_pipeline(
    project_id: str,
    dataset_location: str,
    dataset_id: str,
    table_id: str,
    col_label: str,
    col_training: list):
    
    QUERY = """SELECT * FROM `pb-sandbox-1.oxheart_prototype.heart`"""
    METRICS_NAMES = ["accuracy_score", "f1_score"]
    
    ingest = query_to_table(query=QUERY,
                            table_id=table_id,
                            project_id=project_id,
                            dataset_id=dataset_id,
                            location=dataset_location,
                            query_job_config={'write_disposition': 'WRITE_TRUNCATE'}
                           ).set_display_name("Ingest Data")
    
    # From big query store in GCS
    ingested_dataset = (
                        extract_table_to_gcs(
                            project_id=project_id,
                            dataset_id=dataset_id,
                            table_id=table_id,
                            location=dataset_location,
                        )
                        .after(ingest)
                        .set_display_name("Extract Big Query to GCS")
                    )
    
    # Split data
    spit_data = create_sets(data_input=ingested_dataset.outputs["dataset"],
                              col_label=col_label,
                              col_training=col_training
                           ).set_display_name("Split data")
    
    # Train model
    training_model = train_model(
        training_data=spit_data.outputs['dataset_train']).set_display_name("Train Model")
    
    # Predit model
    predict_data = predict_model(
                test_data=spit_data.outputs['dataset_test'],
                model=training_model.outputs["model"]
            ).set_display_name("Create Predictions")
    
    
    # Evaluate model
    eval_metrics = evaluation_metrics(
        predictions=predict_data.outputs['predictions'],
        dict_keys=spit_data.outputs['dict_keys'],
        metrics_names=METRICS_NAMES,
        ).set_display_name("Evaluation Metrics")


if __name__ == "__main__":
    DATASET_ID = "oxheart_prototype"
    TABLE_ID = "heart_temp"
    COL_LABEL = "class" 
    COL_TRAINING=[
        'age','sex','chest_pain_type','resting_blood_pressure',
        'chol','fasting_blood_sugar','resting_ECG','max_heart_rate',
        'exang','slope','number_vessels_flourosopy','thal','target'
                  ]

    PIPELINE_PARAMS = {"project_id": PROJECT_ID,
                    "dataset_location": LOCATION,
                    "table_id": TABLE_ID,
                    "dataset_id": DATASET_ID,
                    "col_label": COL_LABEL,
                    "col_training": COL_TRAINING}

    compiler.Compiler().compile(
        pipeline_func=oxheart_prototype_pipeline,
        package_path=TEMPLATE_PATH)
    
    aiplatform.init(project=PROJECT_ID, location=LOCATION)

    pipeline_ = aiplatform.pipeline_jobs.PipelineJob(
    enable_caching=ENABLE_CACHING,
    display_name=PIPELINE_NAME,
    template_path=TEMPLATE_PATH,
    job_id=JOBID,
    parameter_values=PIPELINE_PARAMS)

    pipeline_.submit(service_account=SERVICE_ACCOUNT)
