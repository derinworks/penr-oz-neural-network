from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel, Field
from neural_net_model import NeuralNetworkModel
import os

app = FastAPI(
    title="Neural Network Model API",
    description="API to create, serialize, and compute output of neural network models.",
    version="0.0.1"
)

class ActivationRequest(BaseModel):
    activation_vector: list[float] = Field(
        default=...,
        example=[0, 1, 0, 0, 1, 0, 0, 0, 0],
        description="A list of numerical values representing the activation vector."
    )

class CreateModelRequest(BaseModel):
    model_id: str = Field(
        default=...,
        example="test",
        description="The unique identifier for the model."
    )
    input_size: int = Field(
        default=...,
        example=9,
        description="The size of the input layer."
    )
    output_size: int = Field(
        default=...,
        example=9,
        description="The size of the output layer."
    )

@app.post("/model/")
def create_model(body: CreateModelRequest = Body(...)):
    try:
        model = NeuralNetworkModel(body.input_size, body.output_size)
        filepath = f"model_{body.model_id}.json"
        model.serialize(filepath)
        return {"message": "Model created and saved successfully", "file": filepath}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.put("/model/{model_id}/")
def compute_model_output(model_id: str = "test", body: ActivationRequest = Body(...)):
    try:
        filepath = f"model_{model_id}.json"
        model = NeuralNetworkModel.deserialize(filepath)
        output_vector = model.compute_output(body.activation_vector)
        return {"output_vector": output_vector}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
