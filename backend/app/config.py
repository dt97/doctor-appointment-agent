"""
Configuration settings for the Doctor Appointment Booking Agent.

Chain of Thought:
- Load environment variables for API keys
- Define specialist mappings for symptom analysis
- Configure application settings
"""

import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

SPECIALIST_MAPPING = {
    "cardiologist": {
        "keywords": ["chest pain", "heart", "palpitation", "blood pressure", "bp", "cardiac", "heartbeat"],
        "description": "Heart and cardiovascular system specialist"
    },
    "dermatologist": {
        "keywords": ["skin", "rash", "acne", "eczema", "psoriasis", "hair loss", "itching"],
        "description": "Skin, hair, and nail specialist"
    },
    "orthopedic": {
        "keywords": ["bone", "joint", "fracture", "back pain", "spine", "knee", "shoulder", "arthritis"],
        "description": "Bone and joint specialist"
    },
    "neurologist": {
        "keywords": ["headache", "migraine", "seizure", "numbness", "dizziness", "memory", "nerve"],
        "description": "Brain and nervous system specialist"
    },
    "gastroenterologist": {
        "keywords": ["stomach", "digestion", "acidity", "liver", "intestine", "constipation", "diarrhea"],
        "description": "Digestive system specialist"
    },
    "pulmonologist": {
        "keywords": ["breathing", "lungs", "asthma", "cough", "respiratory", "shortness of breath"],
        "description": "Lung and respiratory specialist"
    },
    "ophthalmologist": {
        "keywords": ["eye", "vision", "blurry", "cataract", "glaucoma"],
        "description": "Eye specialist"
    },
    "ent_specialist": {
        "keywords": ["ear", "nose", "throat", "hearing", "sinus", "tonsil"],
        "description": "Ear, Nose, and Throat specialist"
    },
    "psychiatrist": {
        "keywords": ["anxiety", "depression", "stress", "sleep disorder", "mental health", "panic"],
        "description": "Mental health specialist"
    },
    "general_physician": {
        "keywords": ["fever", "cold", "flu", "fatigue", "general", "weakness", "body ache"],
        "description": "General health issues and primary care"
    }
}
