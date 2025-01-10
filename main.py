from fastapi import FastAPI, HTTPException, Body
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


class ActivationRequest(BaseModel):
    model_id: str = Field(
        ...,
        description="The unique identifier for the model."
    )
    input: InputItem = Field(
        ...,
        description="The input data, an InputItem."
    )


class CreateModelRequest(BaseModel):
    model_id: str = Field(
        ...,
        examples=["test"],
        description="The unique identifier for the model."
    )
    layer_sizes: list[int] = Field(
        ...,
        examples=[[9, 9, 9]],
        description="A list of integers representing the sizes of each layer in the neural network."
    )


class TrainingRequest(BaseModel):
    model_id: str = Field(
        ...,
        examples=["test"],
        description="The unique identifier for the model."
    )
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
        model = NeuralNetworkModel(body.layer_sizes)
        filepath = f"model_{body.model_id}.json"
        model.serialize(filepath)
        return {"message": "Model created and saved successfully", "file": filepath}
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
                                     "input": example
                                 }
                             } for idx, example in enumerate(EXAMPLES)} )):
    try:
        filepath = f"model_{body.model_id}.json"
        model = NeuralNetworkModel.deserialize(filepath)
        activation_vector = body.input.activation_vector
        training_vector = body.input.training_vector

        output_vector, cost, cost_derivative_wrt_weights, cost_derivative_wrt_biases = (
            model.compute_output(activation_vector, training_vector))
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
        filepath = f"model_{body.model_id}.json"
        model = NeuralNetworkModel.deserialize(filepath)

        async def train():
            model.train(
                [(data.activation_vector, data.training_vector) for data in body.training_data],
                epochs=body.epochs,
                learning_rate=body.learning_rate
            )
            model.serialize(filepath)

        asyncio.create_task(train())
        return JSONResponse(content={"message": "Training started asynchronously."}, status_code=202)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
