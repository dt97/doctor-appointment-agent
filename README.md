# Doctor Appointment Booking Agent

An AI powered medical appointment booking assistant built with **LangGraph**, **LangChain**, and **FastAPI** for the backend, and **React** for the frontend. Created using `Claude Opus 4.5` model.

## Demo Video

https://github.com/dt97/doctor-appointment-agent/tree/main/videos/doctor_appointment_agent_examples.mov

<video src="videos/doctor_appointment_agent_examples.mov" controls width="100%">
  Your browser does not support the video tag.
</video>

> **Note:** The video demonstrates the complete flow from symptom collection to appointment booking confirmation.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    DOCTOR APPOINTMENT BOOKING AGENT                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  STATE 1: SYMPTOM_COLLECTION                                                │
│  ├── Agent asks user about medical problem                                  │
│  └── User provides symptoms                                                 │
│                                                                             │
│  STATE 2: SYMPTOM_ANALYSIS                                                  │
│  ├── LangChain analyzes symptoms using GPT-4o-mini                          │
│  ├── Maps symptoms → Specialist type                                        │
│  │   ├── Chest pain, BP issues → Cardiologist                               │
│  │   ├── Skin rash, acne → Dermatologist                                    │
│  │   ├── Bone/joint pain → Orthopedic                                       │
│  │   ├── General fever, cold → General Physician                            │
│  │   └── ... other mappings                                                 │
│  └── Agent presents recommendation                                          │
│                                                                             │
│  STATE 3: DOCTOR_CONFIRMATION                                               │
│  ├── Agent asks user to confirm specialist type                             │
│  └── User confirms or requests different specialist                         │
│                                                                             │
│  STATE 4: FETCH_AVAILABILITY                                                │
│  ├── Mock Practo API call to get:                                           │
│  │   ├── Nearby hospitals                                                   │
│  │   ├── Available doctors                                                  │
│  │   └── Time slots                                                         │
│  └── Display options to user via UI                                         │
│                                                                             │
│  STATE 5: SLOT_SELECTION                                                    │
│  ├── User selects doctor, hospital, time slot                               │
│  └── Agent confirms selection                                               │
│                                                                             │
│  STATE 6: BOOKING_CONFIRMATION                                              │
│  ├── Agent shows final booking summary                                      │
│  ├── User confirms booking                                                  │
│  └── Display appointment details with guidelines                            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Chain of Thought - How the Agent Works

### 1. Symptom Collection
- The agent greets the user and asks about their medical concern
- User describes symptoms in natural language

### 2. Symptom Analysis (LangChain + GPT)
- The LLM analyzes the symptom description
- Extracts key symptoms and maps them to specialist types
- Uses a predefined mapping of symptoms → specialists:
  - **Cardiologist**: chest pain, heart, palpitation, blood pressure
  - **Dermatologist**: skin, rash, acne, eczema
  - **Orthopedic**: bone, joint, fracture, back pain
  - **Neurologist**: headache, migraine, seizure, dizziness
  - **Gastroenterologist**: stomach, digestion, acidity, liver
  - **General Physician**: fever, cold, flu, general weakness

### 3. Doctor Confirmation
- Agent presents the recommendation with reasoning
- User can confirm or request a different specialist

### 4. Fetch Availability (Mock Practo API)
- Simulates API call to get nearby hospitals and doctors
- Returns realistic data with:
  - Hospital details (name, address, rating, distance)
  - Doctor details (name, experience, rating, fee)
  - Available time slots for next 3 days

### 5. Slot Selection
- User browses hospitals and doctors in the UI
- Selects preferred doctor and time slot
- UI shows selection summary

### 6. Booking Confirmation
- Agent shows final booking details
- User confirms the appointment
- Agent displays booking ID and guidelines

## Project Structure

```
DoctorAppointmentBookingAgent/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── config.py          # Configuration and specialist mappings
│   │   ├── models.py          # Pydantic models for data structures
│   │   ├── mock_practo_api.py # Simulated Practo API
│   │   ├── langchain_agent.py # Main LangChain agent logic
│   │   └── main.py            # FastAPI application
│   ├── requirements.txt
│   └── .env.example
├── frontend/
│   ├── public/
│   │   └── index.html
│   ├── src/
│   │   ├── App.js             # Main React component
│   │   ├── App.css            # Styles
│   │   ├── index.js           # Entry point
│   │   └── index.css          # Global styles
│   └── package.json
└── README.md
```

## Setup Instructions

### Prerequisites
- Python 3.9+
- Node.js 18+
- OpenAI API Key

### Backend Setup

1. Navigate to backend directory:
   ```bash
   cd backend
   ```

2. Create virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate 
   ```

3. Install dependencies:
   ```bash
   source venv/bin/activate && pip install -r requirements.txt
   ```

4. Create `.env` file:
   ```bash
   cp .env.example .env
   # Edit .env and add your OpenAI API key
   ```

5. Run the backend:
   ```bash
   source venv/bin/activate && uvicorn app.main:app --reload --port 8000
   ```

### Frontend Setup

1. In another terminal tab, navigate to frontend directory:
   ```bash
   cd frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Start the development server:
   ```bash
   npm start
   ```

4. Open http://localhost:3000 in your browser

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/api/session` | POST | Create new session |
| `/api/chat` | POST | Send message to agent |
| `/api/session/{id}` | GET | Get session state |

## Example Conversation Flow

```
Agent: Hello! I'm your medical appointment assistant. Please tell me about 
       your medical concern or symptoms.

User: I have occasional chest pain, and having BP spikes from few days

Agent: I've analyzed your symptoms. Here's what I found:
       
       Identified Symptoms: chest pain, blood pressure spikes
       Recommended Specialist: Cardiologist
       Reason: Chest pain combined with blood pressure issues indicates 
               potential cardiovascular concerns.
       
       Would you like me to find available Cardiologists near you?

User: Yes please

Agent: Great! I found the following Cardiologists near you.
       [Displays list of hospitals and doctors with available slots]

User: [Selects Dr. Rajesh Kumar at Apollo Heart Institute, 10:00 AM tomorrow]

Agent: Please confirm your appointment booking:
       Doctor: Dr. Rajesh Kumar
       Hospital: Apollo Heart Institute
       Date & Time: 2024-01-20 at 10:00 AM
       Fee: ₹800

User: Yes

Agent: 🎉 Appointment Confirmed!
       Booking ID: APT-A1B2C3D4
       
       Important Guidelines:
       • Please arrive 30 minutes before your appointment
       • Carry a valid ID proof
       • Bring previous medical reports
```

## Future Enhancements

1. **Payment Integration**: Add payment gateway for consultation fees
2. **Real Practo API**: Replace mock API with actual Practo integration
3. **User Authentication**: Add login/signup functionality
4. **Appointment History**: Track past and upcoming appointments
5. **Notifications**: SMS/Email reminders for appointments
6. **Multi-language Support**: Support for regional languages

## Technologies Used

- **Backend**: FastAPI, LangChain, OpenAI GPT-4o-mini, Pydantic
- **Frontend**: React, Axios, Lucide Icons
- **Styling**: Custom CSS with modern design

## License

MIT License
