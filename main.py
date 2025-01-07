from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel, Field
from neural_net_model import NeuralNetworkModel

app = FastAPI(
    title="Neural Network Model API",
    description="API to create, serialize, and compute output of neural network models.",
    version="0.0.1"
)

class ActivationRequest(BaseModel):
    model_id: str = Field(
        "test",
        examples=["test"],
        description="The unique identifier for the model."
    )
    activation_vector: list[float] = Field(
        ...,
        examples=[[0, 1, 0, 0, 1, 0, 0, 0, 0]],
        description="A list of numerical values representing the activation vector."
    )
    training_vector: list[float] | None = Field(
        None,
        examples=[[2, 1]],
        description="An optional list of numerical values representing the training vector."
    )

class CreateModelRequest(BaseModel):
    model_id: str = Field(
        ...,
        examples=["test"],
        description="The unique identifier for the model."
    )
    layer_sizes: list[int] = Field(
        ...,
        examples=[[9, 9, 2]],
        description="A list of integers representing the sizes of each layer in the neural network."
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

@app.put("/model/")
def compute_model_output(body: ActivationRequest = Body(...)):
    try:
        filepath = f"model_{body.model_id}.json"
        model = NeuralNetworkModel.deserialize(filepath)
        output_vector, cost = model.compute_output(body.activation_vector, body.training_vector)
        return {"output_vector": output_vector, "cost": cost}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
