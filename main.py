from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import vertexai
from vertexai.language_models import ChatModel
import os
# os.environ['GOOGLE_APPLICATION_CREDENTIALS']='./smart-portfolio-401206-5657d2c792f3.json'
GOOGLE_APPLICATION_CREDENTIALS = os.environ.get('gcp_keys')

vertexai.init(project="smart-portfolio-401206", location="us-central1")
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
    )
# Initialize the chat model
chat_model = ChatModel.from_pretrained("chat-bison")

class ChatInput(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str

@app.post("/chat")
async def chat_with_model(chat_input: ChatInput):
    try:
        parameters = {
            "candidate_count": 1,
            "max_output_tokens": 1024,
            "temperature": 0.2,
            "top_p": 0.8,
            "top_k": 40
        }
        chat = chat_model.start_chat()
        response = chat.send_message(chat_input.message, **parameters)
        return {"response": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in chatbot: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
