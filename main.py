from fastapi import FastAPI, HTTPException, Body
from fastapi.params import Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from neural_net_model import NeuralNetworkModel
import asyncio

app = FastAPI(
    title="Neural Network Model API",
    description="API to create, serialize, and compute output of neural network models.",
    version="0.0.1"
)

# Constants for examples
EXAMPLES = [
    {
        "activation_vector": [0, 0, 0, 0, 2, 0, 0, 0, 0],
        "training_vector": [1., 0, 0, 0, 0, 0, 0, 0, 0]
    },
    {
        "activation_vector": [1, 0, 0, 0, 2, 0, 0, 0, 0],
        "training_vector": [0, 1., 0, 0, 0, 0, 0, 0, 0]
    },
    {
        "activation_vector": [1, 2, 0, 0, 2, 0, 0, 0, 0],
        "training_vector": [0, 0, 0, 0, 0, 0, 0, 1., 0]
    },
    {
        "activation_vector": [1, 2, 0, 0, 2, 0, 0, 1, 0],
        "training_vector": [0, 0, 1., 0, 0, 0, 0, 0, 0]
    },
    {
        "activation_vector": [1, 2, 2, 0, 2, 0, 0, 1, 0],
        "training_vector": [0, 0, 0, 0, 0, 0, 1., 0, 0]
    },
    {
        "activation_vector": [1, 2, 2, 0, 2, 0, 1, 1, 0],
        "training_vector": [0, 0, 0, 0, 0, 0, 0, 0, 1.]
    },
    {
        "activation_vector": [1, 2, 2, 0, 2, 0, 1, 1, 2],
        "training_vector": [0, 0, 0, 1., 0, 0, 0, 0, 0]
    },
    {
        "activation_vector": [1, 2, 2, 1, 2, 0, 1, 1, 2],
        "training_vector": [0, 0, 0, 0, 0, 1., 0, 0, 0]
    },
    {
        "activation_vector": [1, 2, 2, 1, 2, 2, 1, 1, 2],
        "training_vector": [0, 0, 0, 0, 0, 1., 0, 0, 0]
    },
]


class InputItem(BaseModel):
    activation_vector: list[float] = Field(
        ...,
        description="The input activation vector."
    )
    training_vector: list[float] | None = Field(
        None,
        description="The expected output vector, optional for input items."
    )


class TrainingItem(InputItem):
    training_vector: list[float] = Field(
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
    init_algo: str = Field(
        ...,
        examples=["xavier"],
        description="An initialization algorithm"
    )


class ModelMutationRequest(ModelRequest):
    activation_algo: str = Field(
        ...,
        examples=["sigmoid"],
        description="The activation algorithm to apply"
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
        ...,
        examples=[10],
        description="The number of training epochs."
    )
    learning_rate: float = Field(
        ...,
        examples=[0.01],
        description="The learning rate for training."
    )


@app.post("/model/")
def create_model(body: CreateModelRequest = Body(...)):
    try:
        model = NeuralNetworkModel(body.model_id, body.layer_sizes, body.init_algo)
        model.serialize()
        return {"message": f"Model {body.model_id} created and saved successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


@app.post("/output/")
def compute_model_output(body:
    ActivationRequest = Body(...,
                             openapi_examples={f"example_{idx}": {
                                 "summary": f"Example {idx + 1}",
                                 "description": f"Example input and training data for case {idx + 1}",
                                 "value": {
                                     "model_id": "test",
                                     "activation_algo": "sigmoid",
                                     "input": example
                                 }
                             } for idx, example in enumerate(EXAMPLES)} )):
    try:
        model = NeuralNetworkModel.deserialize(body.model_id)
        activation_vector = body.input.activation_vector
        training_vector = body.input.training_vector
        activation_algo = body.activation_algo
        output_vector, cost, cost_derivative_wrt_weights, cost_derivative_wrt_biases = (
            model.compute_output(activation_vector, activation_algo, training_vector))
        return {"output_vector": output_vector,
                "cost": cost,
                "cost_derivative_wrt_weights": cost_derivative_wrt_weights,
                "cost_derivative_wrt_biases": cost_derivative_wrt_biases,
                }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


@app.put("/train/")
async def train_model(body: TrainingRequest = Body(...)):
    try:
        model = NeuralNetworkModel.deserialize(body.model_id)

        async def train():
            model.train(
                [(data.activation_vector, data.training_vector) for data in body.training_data],
                activation_algo= body.activation_algo,
                epochs=body.epochs,
                learning_rate=body.learning_rate,
            )

        asyncio.create_task(train())
        return JSONResponse(content={"message": "Training started asynchronously."}, status_code=202)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


@app.get("/progress/")
def model_progress(model_id: str = Query(..., description="The unique identifier for the model.")):
    try:
        model = NeuralNetworkModel.deserialize(model_id)
        return {
            "progress": model.progress
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
