from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
from openai import AsyncOpenAI
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

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

# Initialize OpenAI client
openai = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Pydantic models for request/response
class VisitDetails(BaseModel):
    date: str
    doctor: str
    department: str
    reason: str
    diagnoses: List[str]
    visitSummary: str

class PatientDetails(BaseModel):
    name: str
    age: int
    gender: str
    conditions: List[str]
    medications: List[str]
    allergies: List[str]
    visits: List[VisitDetails]

class PatientVisit(BaseModel):
    time: str
    patient_id: str
    doctor: str
    department: str
    reason: str
    patientDetails: Optional[PatientDetails] = None

@app.post("/api/generate-day-summary")
async def generate_day_summary(visits: List[PatientVisit]):
    try:
        logger.info(f"Received request with {len(visits)} visits")
        logger.info(f"OpenAI API Key present: {bool(os.getenv('OPENAI_API_KEY'))}")
        
        prompt = f"""As a medical assistant, generate a concise morning briefing for a doctor about their day's schedule. 
Include relevant patient history and important medical information. Be professional but conversational.

IMPORTANT: For each patient, always start by clearly stating their name, age, and gender (e.g., "John Smith, a 45-year-old male" or "Sarah Johnson, a 32-year-old female").

Today's schedule:
{chr(10).join(f'''
Time: {visit.time}
Patient: {visit.patientDetails.name if visit.patientDetails else 'New Patient'}
Reason: {visit.reason}
{chr(10).join(f'''
Medical History:
- Age: {visit.patientDetails.age}
- Gender: {visit.patientDetails.gender}
- Conditions: {', '.join(visit.patientDetails.conditions) or 'None'}
- Medications: {', '.join(visit.patientDetails.medications) or 'None'}
- Allergies: {', '.join(visit.patientDetails.allergies) or 'None'}
- Last Visit: {visit.patientDetails.visits[-1].visitSummary if visit.patientDetails.visits else 'No previous visits'}
''' if visit.patientDetails else 'New patient, no medical history available')}
''' for visit in visits)}

Please provide a natural, conversational summary that the doctor can listen to while preparing for their day. Remember to always start each patient's description with their name, age, and gender."""

        logger.info("Sending request to OpenAI")
        completion = await openai.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a medical assistant providing a morning briefing to a doctor about their day's schedule. Be concise, professional, and highlight important medical information."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model="gpt-4.1-mini",
            temperature=0.7,
            max_tokens=5000,
        )
        logger.info("Received response from OpenAI")

        return {"summary": completion.choices[0].message.content}
    except Exception as e:
        logger.error(f"Error in generate_day_summary: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# Route to test the backend
@app.get("/api/hello")
def read_root():
    openai_api_key = os.getenv("OPENAI_API_KEY")
    return {"message": f"Hello from FastAPI! OpenAI Key: {openai_api_key[:5]}"}
