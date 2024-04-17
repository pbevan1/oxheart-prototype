import json
from typing import NamedTuple
from datetime import datetime
import google.cloud.aiplatform as aiplatform
from kfp.v2.dsl import (compiler, component, Input, Model, Output, Dataset, 
                        Artifact, OutputPath, ClassificationMetrics, 
                        Metrics, InputPath)


GCP_BIGQUERY = "google-cloud-bigquery==2.30.0"
PANDAS = "pandas==1.3.2"
SKLEARN = "scikit-learn==1.0.2"
NUMPY = "numpy==1.21.6"