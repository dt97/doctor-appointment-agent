"""
LangGraph Agent for Doctor Appointment Booking.

Chain of Thought - Why LangGraph?
1. **State Graph**: Explicit workflow states as nodes with typed state
2. **Human-in-the-Loop**: Native support via interrupt_before/interrupt_after
3. **Conditional Routing**: Clean branching logic with conditional edges
4. **Checkpointing**: Built-in persistence for resuming conversations
5. **Visualization**: Can visualize the workflow graph for debugging

Graph Structure:
    ┌─────────────────┐
    │     START       │
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │ symptom_collector│ ◄─── Collects symptoms from user
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │ symptom_analyzer │ ◄─── LLM analyzes symptoms, recommends specialist
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │ specialist_confirmer│ ◄─── INTERRUPT: Wait for user to confirm specialist
    └────────┬────────┘
             │
      ┌──────┴──────┐
      │ confirmed?  │
      └──────┬──────┘
        yes/ \no
           │  └──► back to symptom_collector
           ▼
    ┌─────────────────┐
    │ doctor_fetcher  │ ◄─── Fetch doctors from Practo API
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │ slot_selector   │ ◄─── INTERRUPT: Wait for user to select slot
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │ booking_confirmer│ ◄─── INTERRUPT: Wait for booking confirmation
    └────────┬────────┘
             │
      ┌──────┴──────┐
      │ confirmed?  │
      └──────┬──────┘
        yes/ \no
           │  └──► back to slot_selector
           ▼
    ┌─────────────────┐
    │ booking_creator │ ◄─── Create final booking
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │      END        │
    └─────────────────┘
"""

from typing import TypedDict, Annotated, Optional, List
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
import json
import uuid
from datetime import datetime

from app.config import OPENAI_API_KEY, SPECIALIST_MAPPING
from app.mock_practo_api import get_hospitals_by_specialist


def add_messages(left: List[BaseMessage], right: List[BaseMessage]) -> List[BaseMessage]:
    """Reducer to append messages."""
    if left is None:
        left = []
    if right is None:
        right = []
    return left + right


class AgentState(TypedDict):
    """
    Typed state for the LangGraph workflow.
    
    Chain of Thought:
    - messages: Conversation history (append-only with reducer)
    - current_node: Track which node we're at for the frontend
    - symptoms_text: Raw user input about symptoms
    - analysis: Structured symptom analysis result
    - specialist_confirmed: Whether user confirmed the specialist
    - confirmed_specialist: The specialist type user agreed to
    - hospitals: Available hospitals from Practo API
    - selected_doctor/hospital/slot: User's selections
    - booking_confirmed: Whether user confirmed the booking
    - booking: Final booking details
    - awaiting_input: What type of input we're waiting for
    - response_message: Message to send to user
    """
    messages: Annotated[List[BaseMessage], add_messages]
    current_node: str
    symptoms_text: Optional[str]
    analysis: Optional[dict]
    specialist_confirmed: Optional[bool]
    confirmed_specialist: Optional[str]
    hospitals: Optional[List[dict]]
    selected_doctor_id: Optional[str]
    selected_hospital_id: Optional[str]
    selected_slot_id: Optional[str]
    booking_confirmed: Optional[bool]
    booking: Optional[dict]
    awaiting_input: Optional[str]
    response_message: Optional[str]
    response_type: Optional[str]
    response_data: Optional[dict]


class DoctorAppointmentGraph:
    """
    LangGraph-based agent for doctor appointment booking.
    
    Chain of Thought:
    - Build a state graph with specialized nodes
    - Each node handles one specific task
    - Human-in-the-loop via state checks (awaiting_input)
    - Conditional edges for branching logic
    """
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.3,
            api_key=OPENAI_API_KEY
        )
        self.memory = MemorySaver()
        self.graph = self._build_graph()
        self.sessions: dict[str, dict] = {}
    
    def _build_graph(self) -> StateGraph:
        """
        Build the LangGraph workflow.
        
        Chain of Thought:
        1. Define all nodes (each is a specialized agent/function)
        2. Add edges between nodes
        3. Add conditional edges for branching
        4. Compile with checkpointer for persistence
        """
        workflow = StateGraph(AgentState)
        
        workflow.add_node("symptom_collector", self.symptom_collector_node)
        workflow.add_node("symptom_analyzer", self.symptom_analyzer_node)
        workflow.add_node("specialist_confirmer", self.specialist_confirmer_node)
        workflow.add_node("doctor_fetcher", self.doctor_fetcher_node)
        workflow.add_node("slot_selector", self.slot_selector_node)
        workflow.add_node("booking_confirmer", self.booking_confirmer_node)
        workflow.add_node("booking_creator", self.booking_creator_node)
        
        workflow.add_edge(START, "symptom_collector")
        workflow.add_edge("symptom_collector", "symptom_analyzer")
        workflow.add_edge("symptom_analyzer", "specialist_confirmer")
        
        workflow.add_conditional_edges(
            "specialist_confirmer",
            self._route_after_specialist_confirm,
            {
                "fetch_doctors": "doctor_fetcher",
                "collect_symptoms": "symptom_collector",
                "wait": END
            }
        )
        
        workflow.add_edge("doctor_fetcher", "slot_selector")
        
        workflow.add_conditional_edges(
            "slot_selector",
            self._route_after_slot_select,
            {
                "confirm_booking": "booking_confirmer",
                "wait": END
            }
        )
        
        workflow.add_conditional_edges(
            "booking_confirmer",
            self._route_after_booking_confirm,
            {
                "create_booking": "booking_creator",
                "select_slot": "slot_selector",
                "wait": END
            }
        )
        
        workflow.add_edge("booking_creator", END)
        
        return workflow.compile(checkpointer=self.memory)
    
    def symptom_collector_node(self, state: AgentState) -> dict:
        """
        Node: Collect symptoms from user.
        
        Chain of Thought:
        - If no symptoms yet, ask user to provide them
        - If symptoms provided, pass to next node
        """
        if not state.get("symptoms_text"):
            return {
                "current_node": "symptom_collector",
                "awaiting_input": "symptoms",
                "response_message": "Hello! I'm your medical appointment assistant. Please tell me about your medical concern or symptoms, and I'll help you find the right specialist and book an appointment.",
                "response_type": "text",
                "response_data": None
            }
        
        return {
            "current_node": "symptom_collector",
            "awaiting_input": None
        }
    
    async def symptom_analyzer_node(self, state: AgentState) -> dict:
        """
        Node: Analyze symptoms using LLM.
        
        Chain of Thought:
        1. Take symptoms text from state
        2. Send to LLM with specialist mapping context
        3. Parse structured response
        4. Store analysis in state
        """
        symptoms_text = state.get("symptoms_text", "")
        
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
            
            analysis = json.loads(content.strip())
        except Exception as e:
            analysis = {
                "symptoms": [symptoms_text],
                "recommended_specialist": "general_physician",
                "specialist_description": "General health consultation recommended",
                "confidence": 0.5,
                "reasoning": f"Unable to parse specific symptoms, recommending general consultation."
            }
        
        return {
            "current_node": "symptom_analyzer",
            "analysis": analysis,
            "messages": [AIMessage(content=f"Analyzed symptoms: {analysis['reasoning']}")]
        }
    
    def specialist_confirmer_node(self, state: AgentState) -> dict:
        """
        Node: Ask user to confirm specialist recommendation.
        
        Chain of Thought:
        - Present analysis results to user
        - Wait for confirmation (human-in-the-loop)
        - User can confirm, reject, or request different specialist
        """
        analysis = state.get("analysis", {})
        specialist = analysis.get("recommended_specialist", "general_physician")
        specialist_display = specialist.replace("_", " ").title()
        
        if state.get("specialist_confirmed") is None:
            response_message = f"""I've analyzed your symptoms. Here's what I found:

**Identified Symptoms:** {', '.join(analysis.get('symptoms', []))}

**Recommended Specialist:** {specialist_display}
**Reason:** {analysis.get('reasoning', 'Based on your symptoms')}

{analysis.get('specialist_description', '')}

Would you like me to find available {specialist_display}s near you and book an appointment? Please reply with **Yes** to proceed or let me know if you'd prefer a different specialist."""
            
            return {
                "current_node": "specialist_confirmer",
                "awaiting_input": "specialist_confirmation",
                "response_message": response_message,
                "response_type": "confirmation",
                "response_data": {"analysis": analysis, "awaiting_confirmation": True}
            }
        
        return {"current_node": "specialist_confirmer"}
    
    def _route_after_specialist_confirm(self, state: AgentState) -> str:
        """
        Conditional edge: Route based on specialist confirmation.
        
        Chain of Thought:
        - If awaiting input, stop and wait (END)
        - If confirmed, proceed to fetch doctors
        - If rejected with new specialist, update and fetch
        - If rejected without alternative, go back to symptoms
        """
        if state.get("awaiting_input") == "specialist_confirmation":
            return "wait"
        
        if state.get("specialist_confirmed"):
            return "fetch_doctors"
        
        if state.get("confirmed_specialist"):
            return "fetch_doctors"
        
        return "collect_symptoms"
    
    def doctor_fetcher_node(self, state: AgentState) -> dict:
        """
        Node: Fetch doctors from Practo API.
        
        Chain of Thought:
        - Get confirmed specialist type
        - Call mock Practo API
        - Store hospitals/doctors in state
        - Prepare data for frontend display
        """
        specialist = state.get("confirmed_specialist") or state.get("analysis", {}).get("recommended_specialist", "general_physician")
        
        hospitals = get_hospitals_by_specialist(specialist)
        hospitals_data = [h.model_dump() for h in hospitals]
        
        specialist_display = specialist.replace("_", " ").title()
        
        return {
            "current_node": "doctor_fetcher",
            "hospitals": hospitals_data,
            "confirmed_specialist": specialist,
            "response_message": f"Great! I found the following {specialist_display}s near you. Please select a doctor and time slot that works for you.",
            "response_type": "doctor_selection",
            "response_data": {"hospitals": hospitals_data, "specialist_type": specialist_display},
            "awaiting_input": None
        }
    
    def slot_selector_node(self, state: AgentState) -> dict:
        """
        Node: Handle slot selection.
        
        Chain of Thought:
        - If no selection yet, wait for user input
        - If selection made, validate and proceed
        - Present booking summary for confirmation
        """
        if not all([state.get("selected_doctor_id"), state.get("selected_hospital_id"), state.get("selected_slot_id")]):
            return {
                "current_node": "slot_selector",
                "awaiting_input": "slot_selection",
                "response_message": "Please select a doctor and time slot from the options above.",
                "response_type": "doctor_selection",
                "response_data": {
                    "hospitals": state.get("hospitals", []),
                    "specialist_type": state.get("confirmed_specialist", "").replace("_", " ").title()
                }
            }
        
        selected_hospital = None
        selected_doctor = None
        selected_slot = None
        
        for hospital in state.get("hospitals", []):
            if hospital["hospital_id"] == state.get("selected_hospital_id"):
                selected_hospital = hospital
                for doctor in hospital["doctors"]:
                    if doctor["doctor_id"] == state.get("selected_doctor_id"):
                        selected_doctor = doctor
                        for slot in doctor["available_slots"]:
                            if slot["slot_id"] == state.get("selected_slot_id"):
                                selected_slot = slot
                                break
                        break
                break
        
        if not all([selected_hospital, selected_doctor, selected_slot]):
            return {
                "current_node": "slot_selector",
                "awaiting_input": "slot_selection",
                "response_message": "I couldn't find the selected option. Please try selecting again.",
                "response_type": "doctor_selection",
                "response_data": {
                    "hospitals": state.get("hospitals", []),
                    "specialist_type": state.get("confirmed_specialist", "").replace("_", " ").title()
                }
            }
        
        booking_summary = {
            "doctor": selected_doctor,
            "hospital": {
                "hospital_id": selected_hospital["hospital_id"],
                "name": selected_hospital["name"],
                "address": selected_hospital["address"],
                "rating": selected_hospital["rating"]
            },
            "slot": selected_slot,
            "specialist_type": state.get("confirmed_specialist", "").replace("_", " ").title(),
            "symptoms": state.get("analysis", {}).get("symptoms", [])
        }
        
        response_message = f"""Please confirm your appointment booking:

**Doctor:** {selected_doctor['name']}
**Specialization:** {selected_doctor['specialization']}
**Experience:** {selected_doctor['experience_years']} years
**Rating:** ⭐ {selected_doctor['rating']}
**Consultation Fee:** ₹{selected_doctor['consultation_fee']}

**Hospital:** {selected_hospital['name']}
**Address:** {selected_hospital['address']}

**Appointment:** {selected_slot['date']} at {selected_slot['time']}

Would you like to confirm this booking? Reply **Yes** to confirm or **No** to select a different slot."""
        
        return {
            "current_node": "slot_selector",
            "awaiting_input": None,
            "response_message": response_message,
            "response_type": "booking_confirmation",
            "response_data": {"booking_summary": booking_summary}
        }
    
    def _route_after_slot_select(self, state: AgentState) -> str:
        """Route after slot selection."""
        if state.get("awaiting_input") == "slot_selection":
            return "wait"
        
        if state.get("selected_doctor_id") and state.get("selected_slot_id"):
            return "confirm_booking"
        
        return "wait"
    
    def booking_confirmer_node(self, state: AgentState) -> dict:
        """
        Node: Handle booking confirmation.
        
        Chain of Thought:
        - Wait for user to confirm or reject booking
        - If confirmed, proceed to create booking
        - If rejected, go back to slot selection
        """
        if state.get("booking_confirmed") is None:
            return {
                "current_node": "booking_confirmer",
                "awaiting_input": "booking_confirmation"
            }
        
        return {"current_node": "booking_confirmer"}
    
    def _route_after_booking_confirm(self, state: AgentState) -> str:
        """Route after booking confirmation."""
        if state.get("awaiting_input") == "booking_confirmation":
            return "wait"
        
        if state.get("booking_confirmed") is True:
            return "create_booking"
        
        if state.get("booking_confirmed") is False:
            return "select_slot"
        
        return "wait"
    
    def booking_creator_node(self, state: AgentState) -> dict:
        """
        Node: Create final booking.
        
        Chain of Thought:
        - Generate booking ID
        - Compile all booking details
        - Add guidelines
        - Return final confirmation
        """
        selected_hospital = None
        selected_doctor = None
        selected_slot = None
        
        for hospital in state.get("hospitals", []):
            if hospital["hospital_id"] == state.get("selected_hospital_id"):
                selected_hospital = hospital
                for doctor in hospital["doctors"]:
                    if doctor["doctor_id"] == state.get("selected_doctor_id"):
                        selected_doctor = doctor
                        for slot in doctor["available_slots"]:
                            if slot["slot_id"] == state.get("selected_slot_id"):
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
        
        booking = {
            "booking_id": booking_id,
            "doctor": selected_doctor,
            "hospital": selected_hospital,
            "selected_slot": selected_slot,
            "specialist_type": state.get("confirmed_specialist"),
            "symptoms": state.get("analysis", {}).get("symptoms", []),
            "booking_time": datetime.now().isoformat(),
            "guidelines": guidelines
        }
        
        response_message = f"""🎉 **Appointment Confirmed!**

**Booking ID:** {booking_id}

---

**Doctor:** {selected_doctor['name']}
**Specialization:** {selected_doctor['specialization']}
**Experience:** {selected_doctor['experience_years']} years

**Hospital:** {selected_hospital['name']}
**Address:** {selected_hospital['address']}

**Date & Time:** {selected_slot['date']} at {selected_slot['time']}

**Consultation Fee:** ₹{selected_doctor['consultation_fee']}

---

**Important Guidelines:**
• Please arrive 30 minutes before your appointment time for registration formalities.
• Carry a valid ID proof (Aadhaar/PAN/Driving License).
• Bring any previous medical reports or prescriptions related to your condition.
• If you need to cancel or reschedule, please do so at least 4 hours in advance.
• Wear a mask and follow COVID-19 safety protocols at the hospital.

---

Thank you for using our service! Wishing you good health. 🏥"""
        
        return {
            "current_node": "booking_creator",
            "booking": booking,
            "awaiting_input": None,
            "response_message": response_message,
            "response_type": "booking_complete",
            "response_data": {"booking": booking, "booking_id": booking_id}
        }
    
    async def process_message(self, session_id: str, user_message: str, selected_data: Optional[dict] = None) -> dict:
        """
        Process user message through the graph.
        
        Chain of Thought:
        1. Get or create session state
        2. Update state based on user input and current awaiting_input
        3. Run graph from current state
        4. Return response to frontend
        """
        config = {"configurable": {"thread_id": session_id}}
        
        current_state = self.graph.get_state(config)
        state_values = current_state.values if current_state.values else {}
        
        updates = {"messages": [HumanMessage(content=user_message)]}
        
        awaiting = state_values.get("awaiting_input")
        
        if awaiting == "symptoms" or not state_values:
            updates["symptoms_text"] = user_message
        
        elif awaiting == "specialist_confirmation":
            positive_responses = ["yes", "yeah", "yep", "sure", "ok", "okay", "proceed", "go ahead", "please", "confirm"]
            user_lower = user_message.lower().strip()
            
            if any(pos in user_lower for pos in positive_responses):
                updates["specialist_confirmed"] = True
                updates["confirmed_specialist"] = state_values.get("analysis", {}).get("recommended_specialist")
                updates["awaiting_input"] = None
            else:
                for specialist in SPECIALIST_MAPPING.keys():
                    if specialist.replace("_", " ") in user_lower or specialist in user_lower:
                        updates["specialist_confirmed"] = True
                        updates["confirmed_specialist"] = specialist
                        updates["awaiting_input"] = None
                        break
                else:
                    updates["specialist_confirmed"] = False
                    updates["awaiting_input"] = None
        
        elif awaiting == "slot_selection" and selected_data:
            updates["selected_doctor_id"] = selected_data.get("doctor_id")
            updates["selected_hospital_id"] = selected_data.get("hospital_id")
            updates["selected_slot_id"] = selected_data.get("slot_id")
            updates["awaiting_input"] = None
        
        elif awaiting == "booking_confirmation":
            positive_responses = ["yes", "yeah", "yep", "sure", "ok", "okay", "proceed", "go ahead", "confirm"]
            negative_responses = ["no", "nope", "cancel", "back", "change"]
            user_lower = user_message.lower().strip()
            
            if any(pos in user_lower for pos in positive_responses):
                updates["booking_confirmed"] = True
                updates["awaiting_input"] = None
            elif any(neg in user_lower for neg in negative_responses):
                updates["booking_confirmed"] = False
                updates["selected_doctor_id"] = None
                updates["selected_hospital_id"] = None
                updates["selected_slot_id"] = None
                updates["awaiting_input"] = None
        
        result = await self.graph.ainvoke(updates, config)
        
        return {
            "message": result.get("response_message", "How can I help you?"),
            "state": result.get("current_node", "symptom_collector"),
            "message_type": result.get("response_type", "text"),
            "data": result.get("response_data"),
            "session_id": session_id,
            "awaiting_input": result.get("awaiting_input")
        }
    
    def get_initial_message(self, session_id: str) -> dict:
        """Get initial greeting for new session."""
        return {
            "message": "Hello! I'm your medical appointment assistant. Please tell me about your medical concern or symptoms, and I'll help you find the right specialist and book an appointment.",
            "state": "symptom_collector",
            "message_type": "text",
            "data": None,
            "session_id": session_id,
            "awaiting_input": "symptoms"
        }


graph_agent = DoctorAppointmentGraph()
