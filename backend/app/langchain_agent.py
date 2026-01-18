"""
LangChain Agent for Doctor Appointment Booking.

Chain of Thought - Agent Architecture:
1. The agent uses OpenAI's GPT model for natural language understanding
2. Tools are defined for specific actions:
   - analyze_symptoms: Extracts symptoms and recommends specialist
   - fetch_doctors: Gets available doctors from mock Practo API
   - create_booking: Finalizes the appointment booking
3. The agent maintains conversation context through ConversationState
4. Each workflow state has specific handling logic
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain.tools import tool
from typing import List, Optional
import json

from app.config import OPENAI_API_KEY, SPECIALIST_MAPPING
from app.models import (
    ConversationState, WorkflowState, SymptomAnalysisResult,
    ChatMessage, BookingDetails, Hospital, Doctor, TimeSlot
)
from app.mock_practo_api import get_hospitals_by_specialist
from datetime import datetime
import uuid


class DoctorAppointmentAgent:
    """
    Main agent class that orchestrates the appointment booking workflow.
    
    Chain of Thought:
    - Initialize with OpenAI LLM
    - Maintain session states in memory (can be replaced with Redis/DB)
    - Process messages based on current workflow state
    - Transition between states based on user input and agent decisions
    """
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.3,
            api_key=OPENAI_API_KEY
        )
        self.sessions: dict[str, ConversationState] = {}
    
    def get_or_create_session(self, session_id: str) -> ConversationState:
        """Get existing session or create new one."""
        if session_id not in self.sessions:
            self.sessions[session_id] = ConversationState(session_id=session_id)
            initial_message = ChatMessage(
                role="assistant",
                content="Hello! I'm your medical appointment assistant. Please tell me about your medical concern or symptoms, and I'll help you find the right specialist and book an appointment.",
                message_type="text"
            )
            self.sessions[session_id].messages.append(initial_message)
        return self.sessions[session_id]
    
    async def analyze_symptoms(self, symptoms_text: str) -> SymptomAnalysisResult:
        """
        Use LLM to analyze symptoms and recommend specialist.
        
        Chain of Thought:
        1. Send symptoms to LLM with structured prompt
        2. LLM identifies key symptoms
        3. LLM maps symptoms to specialist type
        4. Return structured analysis result
        """
        specialist_info = "\n".join([
            f"- {name}: {info['description']} (keywords: {', '.join(info['keywords'])})"
            for name, info in SPECIALIST_MAPPING.items()
        ])
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"""You are a medical triage assistant. Analyze the patient's symptoms and recommend the most appropriate specialist.

Available specialists and their areas:
{specialist_info}

Respond in JSON format with these fields:
- symptoms: list of identified symptoms
- recommended_specialist: one of the specialist types listed above (use exact key name like "cardiologist", "general_physician")
- specialist_description: brief description of why this specialist
- confidence: float between 0 and 1
- reasoning: brief explanation of your recommendation

Be conservative - if symptoms are vague or could be multiple things, recommend general_physician first."""),
            ("human", "{symptoms}")
        ])
        
        chain = prompt | self.llm
        response = await chain.ainvoke({"symptoms": symptoms_text})
        
        try:
            content = response.content
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            
            result_dict = json.loads(content.strip())
            return SymptomAnalysisResult(**result_dict)
        except Exception as e:
            return SymptomAnalysisResult(
                symptoms=[symptoms_text],
                recommended_specialist="general_physician",
                specialist_description="General health consultation recommended",
                confidence=0.5,
                reasoning=f"Unable to parse specific symptoms, recommending general consultation. Error: {str(e)}"
            )
    
    async def process_message(self, session_id: str, user_message: str, selected_data: Optional[dict] = None) -> dict:
        """
        Main message processing function.
        
        Chain of Thought - State Machine:
        1. SYMPTOM_COLLECTION: Waiting for user to describe symptoms
           → On input: Transition to SYMPTOM_ANALYSIS
        
        2. SYMPTOM_ANALYSIS: Analyzing symptoms with LLM
           → Auto-transition to DOCTOR_CONFIRMATION with recommendation
        
        3. DOCTOR_CONFIRMATION: Waiting for user to confirm specialist
           → On "yes": Transition to FETCH_AVAILABILITY
           → On "no" or different specialist: Stay or adjust
        
        4. FETCH_AVAILABILITY: Fetching doctors from Practo API
           → Auto-transition to SLOT_SELECTION with doctor list
        
        5. SLOT_SELECTION: Waiting for user to select doctor/slot
           → On selection: Transition to BOOKING_CONFIRMATION
        
        6. BOOKING_CONFIRMATION: Waiting for final confirmation
           → On "yes": Transition to COMPLETED
           → On "no": Go back to SLOT_SELECTION
        
        7. COMPLETED: Show booking details and guidelines
        """
        state = self.get_or_create_session(session_id)
        
        state.messages.append(ChatMessage(
            role="user",
            content=user_message,
            message_type="text"
        ))
        
        if state.current_state == WorkflowState.SYMPTOM_COLLECTION:
            return await self._handle_symptom_collection(state, user_message)
        
        elif state.current_state == WorkflowState.DOCTOR_CONFIRMATION:
            return await self._handle_doctor_confirmation(state, user_message)
        
        elif state.current_state == WorkflowState.SLOT_SELECTION:
            return await self._handle_slot_selection(state, user_message, selected_data)
        
        elif state.current_state == WorkflowState.BOOKING_CONFIRMATION:
            return await self._handle_booking_confirmation(state, user_message, selected_data)
        
        elif state.current_state == WorkflowState.COMPLETED:
            return self._handle_completed(state, user_message)
        
        return self._create_response(state, "I'm not sure how to help with that. Could you please describe your symptoms?")
    
    async def _handle_symptom_collection(self, state: ConversationState, symptoms_text: str) -> dict:
        """
        Handle symptom collection state.
        
        Chain of Thought:
        1. Store the symptom description
        2. Call LLM to analyze symptoms
        3. Store analysis results
        4. Transition to DOCTOR_CONFIRMATION
        5. Ask user to confirm specialist recommendation
        """
        state.symptom_description = symptoms_text
        
        analysis = await self.analyze_symptoms(symptoms_text)
        
        state.symptoms = analysis.symptoms
        state.recommended_specialist = analysis.recommended_specialist
        state.specialist_reasoning = analysis.reasoning
        
        state.current_state = WorkflowState.DOCTOR_CONFIRMATION
        
        specialist_display = analysis.recommended_specialist.replace("_", " ").title()
        
        response_message = f"""I've analyzed your symptoms. Here's what I found:

**Identified Symptoms:** {', '.join(analysis.symptoms)}

**Recommended Specialist:** {specialist_display}
**Reason:** {analysis.reasoning}

{analysis.specialist_description}

Would you like me to find available {specialist_display}s near you and book an appointment? Please reply with **Yes** to proceed or let me know if you'd prefer a different specialist."""
        
        return self._create_response(
            state, 
            response_message,
            message_type="confirmation",
            data={
                "analysis": analysis.model_dump(),
                "awaiting_confirmation": True
            }
        )
    
    async def _handle_doctor_confirmation(self, state: ConversationState, user_message: str) -> dict:
        """
        Handle doctor confirmation state.
        
        Chain of Thought:
        1. Check if user confirmed (yes/proceed/ok) or declined
        2. If confirmed: Fetch doctors and transition to SLOT_SELECTION
        3. If declined: Ask for preferred specialist or re-analyze
        """
        positive_responses = ["yes", "yeah", "yep", "sure", "ok", "okay", "proceed", "go ahead", "please", "confirm"]
        user_lower = user_message.lower().strip()
        
        if any(pos in user_lower for pos in positive_responses):
            state.confirmed_specialist = state.recommended_specialist
            
            hospitals = get_hospitals_by_specialist(state.confirmed_specialist)
            state.available_hospitals = hospitals
            
            state.current_state = WorkflowState.SLOT_SELECTION
            
            specialist_display = state.confirmed_specialist.replace("_", " ").title()
            
            hospitals_data = [h.model_dump() for h in hospitals]
            
            response_message = f"""Great! I found the following {specialist_display}s near you. Please select a doctor and time slot that works for you."""
            
            return self._create_response(
                state,
                response_message,
                message_type="doctor_selection",
                data={
                    "hospitals": hospitals_data,
                    "specialist_type": specialist_display
                }
            )
        else:
            for specialist in SPECIALIST_MAPPING.keys():
                if specialist.replace("_", " ") in user_lower or specialist in user_lower:
                    state.recommended_specialist = specialist
                    state.confirmed_specialist = specialist
                    
                    hospitals = get_hospitals_by_specialist(specialist)
                    state.available_hospitals = hospitals
                    state.current_state = WorkflowState.SLOT_SELECTION
                    
                    specialist_display = specialist.replace("_", " ").title()
                    hospitals_data = [h.model_dump() for h in hospitals]
                    
                    return self._create_response(
                        state,
                        f"Sure, I'll find {specialist_display}s for you. Here are the available options:",
                        message_type="doctor_selection",
                        data={
                            "hospitals": hospitals_data,
                            "specialist_type": specialist_display
                        }
                    )
            
            return self._create_response(
                state,
                "I understand you'd like a different specialist. Could you please tell me which type of specialist you'd prefer, or describe your symptoms again so I can re-analyze?",
                message_type="text"
            )
    
    async def _handle_slot_selection(self, state: ConversationState, user_message: str, selected_data: Optional[dict]) -> dict:
        """
        Handle slot selection state.
        
        Chain of Thought:
        1. User selects doctor, hospital, and time slot via UI
        2. selected_data contains: doctor_id, hospital_id, slot_id
        3. Validate selection exists in available options
        4. Store selection and transition to BOOKING_CONFIRMATION
        5. Show booking summary for final confirmation
        """
        if selected_data and all(k in selected_data for k in ["doctor_id", "hospital_id", "slot_id"]):
            state.selected_doctor_id = selected_data["doctor_id"]
            state.selected_hospital_id = selected_data["hospital_id"]
            state.selected_slot_id = selected_data["slot_id"]
            
            selected_hospital = None
            selected_doctor = None
            selected_slot = None
            
            for hospital in state.available_hospitals:
                if hospital.hospital_id == state.selected_hospital_id:
                    selected_hospital = hospital
                    for doctor in hospital.doctors:
                        if doctor.doctor_id == state.selected_doctor_id:
                            selected_doctor = doctor
                            for slot in doctor.available_slots:
                                if slot.slot_id == state.selected_slot_id:
                                    selected_slot = slot
                                    break
                            break
                    break
            
            if not all([selected_hospital, selected_doctor, selected_slot]):
                return self._create_response(
                    state,
                    "I couldn't find the selected option. Please try selecting again.",
                    message_type="doctor_selection",
                    data={
                        "hospitals": [h.model_dump() for h in state.available_hospitals],
                        "specialist_type": state.confirmed_specialist.replace("_", " ").title()
                    }
                )
            
            state.current_state = WorkflowState.BOOKING_CONFIRMATION
            
            booking_summary = {
                "doctor": selected_doctor.model_dump(),
                "hospital": {
                    "hospital_id": selected_hospital.hospital_id,
                    "name": selected_hospital.name,
                    "address": selected_hospital.address,
                    "rating": selected_hospital.rating
                },
                "slot": selected_slot.model_dump(),
                "specialist_type": state.confirmed_specialist.replace("_", " ").title(),
                "symptoms": state.symptoms
            }
            
            response_message = f"""Please confirm your appointment booking:

**Doctor:** {selected_doctor.name}
**Specialization:** {selected_doctor.specialization}
**Experience:** {selected_doctor.experience_years} years
**Rating:** ⭐ {selected_doctor.rating}
**Consultation Fee:** ₹{selected_doctor.consultation_fee}

**Hospital:** {selected_hospital.name}
**Address:** {selected_hospital.address}

**Appointment:** {selected_slot.date} at {selected_slot.time}

Would you like to confirm this booking? Reply **Yes** to confirm or **No** to select a different slot."""
            
            return self._create_response(
                state,
                response_message,
                message_type="booking_confirmation",
                data={"booking_summary": booking_summary}
            )
        else:
            return self._create_response(
                state,
                "Please select a doctor and time slot from the options above.",
                message_type="doctor_selection",
                data={
                    "hospitals": [h.model_dump() for h in state.available_hospitals],
                    "specialist_type": state.confirmed_specialist.replace("_", " ").title()
                }
            )
    
    async def _handle_booking_confirmation(self, state: ConversationState, user_message: str, selected_data: Optional[dict]) -> dict:
        """
        Handle booking confirmation state.
        
        Chain of Thought:
        1. Check if user confirmed or cancelled
        2. If confirmed: Create booking, transition to COMPLETED
        3. If cancelled: Go back to SLOT_SELECTION
        4. Show final booking details with guidelines
        """
        positive_responses = ["yes", "yeah", "yep", "sure", "ok", "okay", "proceed", "go ahead", "confirm"]
        negative_responses = ["no", "nope", "cancel", "back", "change"]
        user_lower = user_message.lower().strip()
        
        if any(neg in user_lower for neg in negative_responses):
            state.current_state = WorkflowState.SLOT_SELECTION
            return self._create_response(
                state,
                "No problem! Please select a different doctor or time slot:",
                message_type="doctor_selection",
                data={
                    "hospitals": [h.model_dump() for h in state.available_hospitals],
                    "specialist_type": state.confirmed_specialist.replace("_", " ").title()
                }
            )
        
        if any(pos in user_lower for pos in positive_responses):
            selected_hospital = None
            selected_doctor = None
            selected_slot = None
            
            for hospital in state.available_hospitals:
                if hospital.hospital_id == state.selected_hospital_id:
                    selected_hospital = hospital
                    for doctor in hospital.doctors:
                        if doctor.doctor_id == state.selected_doctor_id:
                            selected_doctor = doctor
                            for slot in doctor.available_slots:
                                if slot.slot_id == state.selected_slot_id:
                                    selected_slot = slot
                                    break
                            break
                    break
            
            booking_id = f"APT-{uuid.uuid4().hex[:8].upper()}"
            
            guidelines = [
                "Please arrive 30 minutes before your appointment time for registration formalities.",
                "Carry a valid ID proof (Aadhaar/PAN/Driving License).",
                "Bring any previous medical reports or prescriptions related to your condition.",
                "If you need to cancel or reschedule, please do so at least 4 hours in advance.",
                "Wear a mask and follow COVID-19 safety protocols at the hospital."
            ]
            
            booking = BookingDetails(
                booking_id=booking_id,
                doctor=selected_doctor,
                hospital=selected_hospital,
                selected_slot=selected_slot,
                specialist_type=state.confirmed_specialist,
                symptoms=state.symptoms,
                booking_time=datetime.now(),
                guidelines=guidelines
            )
            
            state.booking = booking
            state.current_state = WorkflowState.COMPLETED
            
            response_message = f"""🎉 **Appointment Confirmed!**

**Booking ID:** {booking_id}

---

**Doctor:** {selected_doctor.name}
**Specialization:** {selected_doctor.specialization}
**Experience:** {selected_doctor.experience_years} years

**Hospital:** {selected_hospital.name}
**Address:** {selected_hospital.address}

**Date & Time:** {selected_slot.date} at {selected_slot.time}

**Consultation Fee:** ₹{selected_doctor.consultation_fee}

---

**Important Guidelines:**
• Please arrive 30 minutes before your appointment time for registration formalities.
• Carry a valid ID proof (Aadhaar/PAN/Driving License).
• Bring any previous medical reports or prescriptions related to your condition.
• If you need to cancel or reschedule, please do so at least 4 hours in advance.
• Wear a mask and follow COVID-19 safety protocols at the hospital.

---

Thank you for using our service! Wishing you good health. 🏥"""
            
            return self._create_response(
                state,
                response_message,
                message_type="booking_complete",
                data={"booking": booking.model_dump(), "booking_id": booking_id}
            )
        
        return self._create_response(
            state,
            "Please confirm your booking by replying **Yes** or **No** to go back and select a different option.",
            message_type="booking_confirmation"
        )
    
    def _handle_completed(self, state: ConversationState, user_message: str) -> dict:
        """
        Handle completed state - allow starting new booking.
        """
        new_booking_keywords = ["new", "another", "different", "book", "appointment"]
        user_lower = user_message.lower()
        
        if any(kw in user_lower for kw in new_booking_keywords):
            state.current_state = WorkflowState.SYMPTOM_COLLECTION
            state.symptoms = []
            state.symptom_description = None
            state.recommended_specialist = None
            state.confirmed_specialist = None
            state.available_hospitals = []
            state.selected_doctor_id = None
            state.selected_hospital_id = None
            state.selected_slot_id = None
            state.booking = None
            
            return self._create_response(
                state,
                "Sure! Let's book a new appointment. Please tell me about your medical concern or symptoms.",
                message_type="text"
            )
        
        return self._create_response(
            state,
            f"Your appointment (Booking ID: {state.booking.booking_id}) is confirmed. Is there anything else I can help you with? Say 'new appointment' to book another appointment.",
            message_type="text"
        )
    
    def _create_response(self, state: ConversationState, message: str, message_type: str = "text", data: Optional[dict] = None) -> dict:
        """Create standardized response and update conversation history."""
        state.messages.append(ChatMessage(
            role="assistant",
            content=message,
            message_type=message_type,
            data=data
        ))
        
        return {
            "message": message,
            "state": state.current_state.value,
            "message_type": message_type,
            "data": data,
            "session_id": state.session_id
        }
    
    def get_initial_message(self, session_id: str) -> dict:
        """Get initial greeting message for new session."""
        state = self.get_or_create_session(session_id)
        return {
            "message": state.messages[0].content if state.messages else "Hello! Please tell me about your medical concern.",
            "state": state.current_state.value,
            "message_type": "text",
            "data": None,
            "session_id": session_id
        }


agent = DoctorAppointmentAgent()
