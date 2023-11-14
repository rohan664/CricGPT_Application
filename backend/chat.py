from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import vertexai
from langchain.llms import VertexAI
import os
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.vectorstores.neo4j_vector import Neo4jVector
from langchain.embeddings import VertexAIEmbeddings
from langchain.memory import ConversationBufferWindowMemory



# os.environ['GOOGLE_APPLICATION_CREDENTIALS']='./smart-portfolio-401206-5657d2c792f3.json'
GOOGLE_APPLICATION_CREDENTIALS = os.environ.get('gcp_keys')

vertexai.init(project="smart-portfolio-401206", location="us-central1")
app = FastAPI()
template = """
You are a helpfull cricket coach assistant which helps in strategies and other decision of palyers, you should only give answers to the question related to cricket only and if other irrelevant questions to cricket  asked then clearly tell them i am only responsible for assisting cricket related decision or information
you should only give answers in english language
------
<ctx>
{context}
</ctx>
------
<hs>
{history}
</hs>
------
{question}
Answer:
"""
prompt = PromptTemplate(
    input_variables=["history", "context", "question"],
    template=template,
)
url = os.environ.get('url')
username = os.environ.get('username')
password = os.environ.get('password')
neo4j_vector = Neo4jVector.from_existing_index(
    VertexAIEmbeddings(),
    url=url,
    username=username,
    password=password,
    index_name="cric-gpt",
    text_node_property="info",  # Need to define if it is not default
)
qa = RetrievalQA.from_chain_type(
    llm=VertexAI(),
    chain_type='stuff',
    retriever=neo4j_vector.as_retriever(search_kwargs={'k': 6}),
    verbose=True,
    chain_type_kwargs={
        "verbose": True,
        "prompt": prompt,
        "memory": ConversationBufferWindowMemory(
            k=4,
            memory_key="history",
            input_key="question"),
    }
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
    )

class ChatInput(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str

@app.post("/chat")
async def chat_with_model(chat_input: ChatInput):
    try:
        response=qa.run(chat_input.message)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in chatbot: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
