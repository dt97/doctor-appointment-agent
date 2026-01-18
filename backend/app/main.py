"""
FastAPI Backend for Doctor Appointment Booking Agent.

Chain of Thought:
- Expose REST API endpoints for the frontend
- /chat: Main endpoint for processing user messages
- /session: Get or create a new session
- CORS enabled for frontend communication
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uuid
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from app.langgraph_agent import graph_agent as agent
from app.models import UserMessage, AgentResponse

app = FastAPI(
    title="Doctor Appointment Booking Agent",
    description="AI-powered medical appointment booking assistant using LangChain",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    session_id: str
    selected_data: Optional[dict] = None


class SessionRequest(BaseModel):
    session_id: Optional[str] = None


@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "healthy", "service": "Doctor Appointment Booking Agent"}


@app.post("/api/session")
async def create_session(request: SessionRequest = None):
    """
    Create or retrieve a session.
    
    Chain of Thought:
    - If session_id provided, return existing session state
    - If not, create new session with UUID
    - Return initial greeting message
    """
    session_id = request.session_id if request and request.session_id else str(uuid.uuid4())
    
    response = agent.get_initial_message(session_id)
    
    return response


@app.post("/api/chat")
async def chat(request: ChatRequest):
    """
    Process user message and return agent response.
    
    Chain of Thought:
    - Validate session exists
    - Pass message to LangChain agent
    - Agent processes based on current workflow state
    - Return response with updated state and any data
    """
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    
    try:
        response = await agent.process_message(
            session_id=request.session_id,
            user_message=request.message,
            selected_data=request.selected_data
        )
        return response
    except Exception as e:
        logger.exception(f"Error processing message for session {request.session_id}")
        raise HTTPException(status_code=500, detail=f"Error processing message: {str(e)}")


@app.get("/api/session/{session_id}")
async def get_session(session_id: str):
    """
    Get current session state and conversation history.
    """
    if session_id in agent.sessions:
        state = agent.sessions[session_id]
        return {
            "session_id": session_id,
            "current_state": state.current_state.value,
            "messages": [msg.model_dump() for msg in state.messages],
            "symptoms": state.symptoms,
            "recommended_specialist": state.recommended_specialist,
            "confirmed_specialist": state.confirmed_specialist
        }
    else:
        raise HTTPException(status_code=404, detail="Session not found")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
