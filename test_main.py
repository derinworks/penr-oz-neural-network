import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from main import app


client = TestClient(app)


@pytest.fixture
def mock_new_model():
    with patch("main.NeuralNetworkModel") as MockModel:
        mock_instance = MagicMock()
        MockModel.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_deserialized_model():
    with patch("neural_net_model.NeuralNetworkModel.deserialize") as mock_deserialize:
        mock_instance = MagicMock()
        # MockModel.return_value = mock_instance
        mock_deserialize.return_value = mock_instance
        yield mock_instance


def test_model_endpoint(mock_new_model):
    payload = {
        "model_id": "test",
        "layer_sizes": [9, 9, 9],
        "init_algo": "xavier"
    }

    response = client.post("/model/", json=payload)

    assert response.status_code == 200

    assert response.json() == {
        "message": "Model test created and saved successfully"
    }

    mock_new_model.serialize.assert_called_once()


def test_output_endpoint(mock_deserialized_model):
    mock_deserialized_model.compute_output.return_value = [0, 1, 0], None, None, None

    payload = {
        "model_id": "test",
        "activation_algo": "sigmoid",
        "input": {
            "activation_vector": [0, 0, 0]
        }
    }

    response = client.post("/output/", json=payload)

    assert response.json() == {
        "output_vector": [0, 1, 0],
        "cost": None,
        "cost_derivative_wrt_weights": None,
        "cost_derivative_wrt_biases": None
    }

    assert response.status_code == 200


def test_train_endpoint(mock_deserialized_model):
    payload = {
        "model_id": "test",
        "activation_algo": "sigmoid",
        "training_data": [
            {
                "activation_vector": [0, 0, 0],
                "training_vector": [0, 1, 0]
            }
        ],
        "epochs": 2,
        "learning_rate": .01
    }

    response = client.put("/train/", json=payload)

    assert response.status_code == 202

    assert response.json() == {
        "message": "Training started asynchronously."
    }

    mock_deserialized_model.train.assert_called_once()


def test_progress_endpoint(mock_deserialized_model):
    mock_deserialized_model.progress = [
        "Some progress"
    ]

    response = client.get("/progress/", params={"model_id": "test"})

    assert response.status_code == 200

    assert response.json() == {
        "progress": [
            "Some progress"
        ]
    }
