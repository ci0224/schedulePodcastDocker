from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os

# Create FastAPI app
app = FastAPI()

# Allow frontend dev server access (change * to your frontend URL in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # This will allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Route to test the backend
@app.get("/api/hello")
def read_root():
    openai_api_key = os.getenv("OPENAI_API_KEY")
    return {"message": f"Hello from FastAPI! OpenAI Key: {openai_api_key[:5]}"}
