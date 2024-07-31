from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import torch

app = FastAPI()

# Define the input and output models
class Instruction(BaseModel):
    text: str

class Frame(BaseModel):
    data: List[int]  # Assuming frame data is a list of integers (e.g., pixel values)

class InferenceRequest(BaseModel):
    instructions: Instruction
    frames: List[Frame]

class ActionSequence(BaseModel):
    actions: List[str]

# Load your transformer model (replace this with your actual model loading code)
class TransformerModel:
    def __init__(self):
        # Replace with actual model loading code
        self.model = torch.nn.Transformer()

    def generate_sequence(self, instructions, frames):
        # Replace with your actual sequence generation logic
        return ["action1", "action2", "action3"]

model = TransformerModel()

@app.post("/generate", response_model=ActionSequence)
async def generate_sequence(request: InferenceRequest):
    try:
        instructions = request.instructions.text
        frames = [frame.data for frame in request.frames]
        
        # Generate the action sequence
        actions = model.generate_sequence(instructions, frames)
        
        return ActionSequence(actions=actions)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
