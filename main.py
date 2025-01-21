import logging

from fastapi import FastAPI, HTTPException, Body, Request
from fastapi.params import Query
from fastapi.responses import JSONResponse, RedirectResponse, Response
from pydantic import BaseModel, Field
from neural_net_model import NeuralNetworkModel
import asyncio

app = FastAPI(
    title="Neural Network Model API",
    description="API to create, serialize, and compute output of neural network models.",
    version="0.0.1"
)

LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%Y-%m-%dT%H:%M:%S"
logging.basicConfig(
    datefmt=DATE_FORMAT,
    format=LOG_FORMAT,
)
log = logging.getLogger(__name__)


# Constants for examples
EXAMPLES = [
    {
        "activation_vector": [0, 0, 0, 0, 0, 0, 0, 0, 0],
        "target_vector":     [0, 0, 0, 0, 1, 0, 0, 0, 0]
    },
    {
        "activation_vector": [0, 0, 0, 0,.5, 0, 0, 0, 0],
        "target_vector":     [1, 0, 0, 0, 0, 0, 0, 0, 0]
    },
    {
        "activation_vector": [1, 0, 0, 0,.5, 0, 0, 0, 0],
        "target_vector":     [0, 1, 0, 0, 0, 0, 0, 0, 0]
    },
    {
        "activation_vector": [1,.5, 0, 0,.5, 0, 0, 0, 0],
        "target_vector":     [0, 0, 0, 0, 0, 0, 0, 1, 0]
    },
    {
        "activation_vector": [1,.5, 0, 0,.5, 0, 0, 1, 0],
        "target_vector":     [0, 0, 1, 0, 0, 0, 0, 0, 0]
    },
    {
        "activation_vector": [1,.5,.5, 0,.5, 0, 0, 1, 0],
        "target_vector":     [0, 0, 0, 0, 0, 0, 1, 0, 0]
    },
    {
        "activation_vector": [1,.5,.5, 0,.5, 0, 1, 1, 0],
        "target_vector":     [0, 0, 0, 0, 0, 0, 0, 0, 1]
    },
    {
        "activation_vector": [1,.5,.5, 0,.5, 0, 1, 1,.5],
        "target_vector":     [0, 0, 0, 1, 0, 0, 0, 0, 0]
    },
    {
        "activation_vector": [1,.5,.5, 1,.5, 0, 1, 1,.5],
        "target_vector":     [0, 0, 0, 0, 0, 1, 0, 0, 0]
    },
]


class InputItem(BaseModel):
    activation_vector: list[float] = Field(
        ...,
        description="The input activation vector."
    )
    target_vector: list[float] | None = Field(
        None,
        description="The expected output vector, optional for input items."
    )


class TrainingItem(InputItem):
    target_vector: list[float] = Field(
        ...,
        description="The expected output vector, required for training items."
    )


class ModelRequest(BaseModel):
    model_id: str = Field(
        ...,
        examples=["test"],
        description="The unique identifier for the model."
    )


class CreateModelRequest(ModelRequest):
    layer_sizes: list[int] = Field(
        ...,
        examples=[[9, 9, 9]],
        description="A list of integers representing the sizes of each layer in the neural network."
    )
    weight_algo: str = Field(
        "xavier",
        examples=["xavier", "he", "gaussian"],
        description="An initialization algorithm"
    )
    bias_algo: str = Field(
        "random",
        examples=["random", "zeros"],
        description="An initialization algorithm"
    )


class ModelMutationRequest(ModelRequest):
    activation_algos: list[str] = Field(
        ...,
        examples=[["sigmoid"] * 2, ["relu"] * 2, ["tanh"] * 2, ["relu", "softmax"]],
        description="The activation algorithms to apply"
    )


class ActivationRequest(ModelMutationRequest):
    input: InputItem = Field(
        ...,
        description="The input data, an InputItem."
    )


class TrainingRequest(ModelMutationRequest):
    training_data: list[TrainingItem] = Field(
        ...,
        examples=[EXAMPLES],
        description="A list of training data pairs."
    )
    epochs: int = Field(
        10,
        examples=[10],
        description="The number of training epochs."
    )
    learning_rate: float = Field(
        0.01,
        examples=[0.01],
        description="The learning rate for training."
    )


class ModelIdQuery(Query):
    description="The unique identifier for the model."


@app.exception_handler(Exception)
async def generic_exception_handler(_: Request, e: Exception):
    log.error(f"An error occurred: {str(e)}")
    return JSONResponse(status_code=500, content={"detail": "Please refer to server logs"})


@app.exception_handler(KeyError)
async def key_error_handler(_: Request, e: KeyError):
    raise HTTPException(status_code=404, detail=f"Not found error occurred: {str(e)}")


@app.exception_handler(ValueError)
async def value_error_handler(_: Request, e: ValueError):
    raise HTTPException(status_code=400, detail=f"Value error occurred: {str(e)}")

@app.get("/", include_in_schema=False)
def redirect_to_docs():
    return RedirectResponse(url="/docs")

@app.post("/model/")
def create_model(body: CreateModelRequest = Body(...)):
    model = NeuralNetworkModel(body.model_id, body.layer_sizes, body.weight_algo, body.bias_algo)
    model.serialize()
    return {"message": f"Model {body.model_id} created and saved successfully"}


@app.post("/output/")
def compute_model_output(body:
    ActivationRequest = Body(...,
                             openapi_examples={f"example_{idx}": {
                                 "summary": f"Example {idx + 1}",
                                 "description": f"Example input and training data for case {idx + 1}",
                                 "value": {
                                     "model_id": "test",
                                     "activation_algos": ["sigmoid"] * 2,
                                     "input": example
                                 }
                             } for idx, example in enumerate(EXAMPLES)} )):
    model = NeuralNetworkModel.deserialize(body.model_id)
    activation_vector = body.input.activation_vector
    target_vector = body.input.target_vector
    activation_algos = body.activation_algos
    output_vector, cost, cost_derivative_wrt_weights, cost_derivative_wrt_biases = (
        model.compute_output(activation_vector, activation_algos, target_vector))
    return {"output_vector": output_vector,
            "cost": cost,
            "cost_derivative_wrt_weights": cost_derivative_wrt_weights,
            "cost_derivative_wrt_biases": cost_derivative_wrt_biases,
            }


@app.put("/train/")
async def train_model(body: TrainingRequest = Body(...)):
    model = NeuralNetworkModel.deserialize(body.model_id)

    async def train():
        model.train(
            [(data.activation_vector, data.target_vector) for data in body.training_data],
            activation_algos= body.activation_algos,
            epochs=body.epochs,
            learning_rate=body.learning_rate,
        )

    asyncio.create_task(train())
    return JSONResponse(content={"message": "Training started asynchronously."}, status_code=202)


@app.get("/progress/")
def model_progress(model_id: str = ModelIdQuery(...)):
    model = NeuralNetworkModel.deserialize(model_id)
    return {
        "progress": model.progress
    }


@app.delete("/model/")
def delete_model(model_id: str = ModelIdQuery(...)):
    NeuralNetworkModel.delete(model_id)
    return Response(status_code=204)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app,
                host="127.0.0.1",
                port=8000,
                log_config={
                    "version": 1,
                    "disable_existing_loggers": False,
                    "formatters": {
                        "default": {
                            "format": LOG_FORMAT,
                            "datefmt": DATE_FORMAT,
                        },
                    },
                    "handlers": {
                        "default": {
                            "level": "INFO",
                            "class": "logging.StreamHandler",
                            "formatter": "default",
                        },
                    },
                    "loggers": {
                        "uvicorn": {
                            "level": "INFO",
                            "handlers": ["default"],
                            "propagate": False,
                        },
                        "uvicorn.error": {
                            "level": "INFO",
                            "handlers": ["default"],
                            "propagate": False,
                        },
                        "uvicorn.access": {
                            "level": "INFO",
                            "handlers": ["default"],
                            "propagate": False,
                        },
                    },
                })
