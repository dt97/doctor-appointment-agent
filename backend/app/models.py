"""
Pydantic models for request/response schemas.

Chain of Thought:
- Define clear data structures for each workflow state
- ChatMessage: Represents a single message in the conversation
- ConversationState: Tracks the current state of the booking workflow
- Doctor, Hospital, TimeSlot: Represent Practo API response data
- BookingDetails: Final appointment confirmation data
"""

from pydantic import BaseModel
from typing import Optional, List
from enum import Enum
from datetime import datetime


class WorkflowState(str, Enum):
    """
    Workflow states for the appointment booking process.
    
    Flow: SYMPTOM_COLLECTION → SYMPTOM_ANALYSIS → DOCTOR_CONFIRMATION 
          → FETCH_AVAILABILITY → SLOT_SELECTION → BOOKING_CONFIRMATION → COMPLETED
    """
    SYMPTOM_COLLECTION = "symptom_collection"
    SYMPTOM_ANALYSIS = "symptom_analysis"
    DOCTOR_CONFIRMATION = "doctor_confirmation"
    FETCH_AVAILABILITY = "fetch_availability"
    SLOT_SELECTION = "slot_selection"
    BOOKING_CONFIRMATION = "booking_confirmation"
    COMPLETED = "completed"


class ChatMessage(BaseModel):
    """Single message in the conversation."""
    role: str  # "user" or "assistant"
    content: str
    message_type: Optional[str] = "text"  # "text", "options", "booking_summary"
    data: Optional[dict] = None  # Additional data like doctor list, slots, etc.


class UserMessage(BaseModel):
    """Incoming message from user."""
    message: str
    session_id: str
    selected_data: Optional[dict] = None  # For slot selection, confirmation, etc.


class TimeSlot(BaseModel):
    """Available appointment time slot."""
    slot_id: str
    time: str
    date: str
    available: bool = True


class Doctor(BaseModel):
    """Doctor information from Practo-like API."""
    doctor_id: str
    name: str
    specialization: str
    experience_years: int
    rating: float
    consultation_fee: int
    hospital_id: str
    available_slots: List[TimeSlot]
    profile_image: Optional[str] = None


class Hospital(BaseModel):
    """Hospital information."""
    hospital_id: str
    name: str
    address: str
    distance_km: float
    rating: float
    doctors: List[Doctor]


class SymptomAnalysisResult(BaseModel):
    """Result of symptom analysis by LangChain agent."""
    symptoms: List[str]
    recommended_specialist: str
    specialist_description: str
    confidence: float
    reasoning: str


class BookingDetails(BaseModel):
    """Final booking confirmation details."""
    booking_id: str
    patient_name: Optional[str] = None
    doctor: Doctor
    hospital: Hospital
    selected_slot: TimeSlot
    specialist_type: str
    symptoms: List[str]
    booking_time: datetime
    guidelines: List[str]


class ConversationState(BaseModel):
    """
    Tracks the entire conversation state for a session.
    
    Chain of Thought:
    - session_id: Unique identifier for this conversation
    - current_state: Which workflow state we're in
    - messages: Full conversation history
    - symptoms: Extracted symptoms from user input
    - recommended_specialist: LLM's recommendation
    - confirmed_specialist: User-confirmed specialist type
    - available_hospitals: Fetched from mock Practo API
    - selected_doctor/hospital/slot: User's selections
    - booking: Final booking details
    """
    session_id: str
    current_state: WorkflowState = WorkflowState.SYMPTOM_COLLECTION
    messages: List[ChatMessage] = []
    symptoms: List[str] = []
    symptom_description: Optional[str] = None
    recommended_specialist: Optional[str] = None
    specialist_reasoning: Optional[str] = None
    confirmed_specialist: Optional[str] = None
    available_hospitals: List[Hospital] = []
    selected_doctor_id: Optional[str] = None
    selected_hospital_id: Optional[str] = None
    selected_slot_id: Optional[str] = None
    booking: Optional[BookingDetails] = None


class AgentResponse(BaseModel):
    """Response from the agent to the frontend."""
    message: str
    state: WorkflowState
    message_type: str = "text"
    data: Optional[dict] = None
    session_id: str
