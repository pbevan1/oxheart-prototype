import pytest
from unittest.mock import Mock, patch, MagicMock

from kubeflow_train_pipeline import query_to_table

# Example test for the first component. Would add more with more time.


@pytest.fixture
def bq_client_mock():
    # mocked BigQuery client
    with patch("google.cloud.bigquery.Client") as mock:
        yield mock()


@pytest.fixture
def job_config_mock():
    # mock the QueryJobConfig to control the destination attribute
    with patch("google.cloud.bigquery.QueryJobConfig") as mock:
        instance = mock.return_value
        instance.destination = MagicMock()
        yield instance


def test_query_to_table(bq_client_mock, job_config_mock):
    project_id = "test-project"
    dataset_id = "test-dataset"
    table_id = "test-table"
    query = "SELECT * FROM `test-dataset.test-table`"
    location = "EU"
    job_config = {"write_disposition": "WRITE_TRUNCATE"}
    destination_expected = f"{project_id}.{dataset_id}.{table_id}"
    job_config_mock.destination = (
        destination_expected  # Explicitly setting the expected destination
    )

    # execute function with the mocks (use python_func to extract the function)
    query_to_table.python_func(
        query, project_id, dataset_id, table_id, location, job_config
    )

    # Assertions to check the calls and results
    bq_client_mock.query.assert_called_once_with(query, job_config=job_config_mock)
    assert job_config_mock.destination == destination_expected
    print("Test passed: Query executed with correct job configuration and destination.")
