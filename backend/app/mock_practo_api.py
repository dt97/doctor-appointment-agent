"""
Mock Practo API for fetching hospitals, doctors, and available slots.

Chain of Thought:
- Since we don't have actual Practo API access, we create realistic mock data
- Data is organized by specialist type
- Each hospital has multiple doctors with available time slots
- This can be replaced with actual API calls in production
"""

from typing import List
from datetime import datetime, timedelta
from app.models import Hospital, Doctor, TimeSlot
import random
import uuid


def generate_time_slots(doctor_id: str, days_ahead: int = 3) -> List[TimeSlot]:
    """
    Generate realistic time slots for the next few days.
    
    Chain of Thought:
    - Create slots for morning (9 AM - 12 PM) and evening (4 PM - 8 PM)
    - Some slots randomly marked as unavailable to simulate real booking
    """
    slots = []
    base_date = datetime.now()
    
    morning_times = ["09:00 AM", "09:30 AM", "10:00 AM", "10:30 AM", "11:00 AM", "11:30 AM"]
    evening_times = ["04:00 PM", "04:30 PM", "05:00 PM", "05:30 PM", "06:00 PM", "06:30 PM", "07:00 PM"]
    
    for day in range(1, days_ahead + 1):
        date = base_date + timedelta(days=day)
        date_str = date.strftime("%Y-%m-%d")
        
        for time in morning_times + evening_times:
            slots.append(TimeSlot(
                slot_id=f"{doctor_id}_{date_str}_{time.replace(' ', '_').replace(':', '')}",
                time=time,
                date=date_str,
                available=random.random() > 0.3  # 70% slots available
            ))
    
    return slots


MOCK_HOSPITALS_DATA = {
    "cardiologist": [
        {
            "hospital_id": "hosp_001",
            "name": "Apollo Heart Institute",
            "address": "Jubilee Hills, Hyderabad",
            "distance_km": 2.5,
            "rating": 4.8,
            "doctors": [
                {"name": "Dr. Rajesh Kumar", "experience": 15, "rating": 4.9, "fee": 800},
                {"name": "Dr. Priya Sharma", "experience": 12, "rating": 4.7, "fee": 700},
            ]
        },
        {
            "hospital_id": "hosp_002",
            "name": "Care Hospitals",
            "address": "Banjara Hills, Hyderabad",
            "distance_km": 4.2,
            "rating": 4.6,
            "doctors": [
                {"name": "Dr. Suresh Reddy", "experience": 20, "rating": 4.8, "fee": 1000},
                {"name": "Dr. Anita Desai", "experience": 8, "rating": 4.5, "fee": 600},
            ]
        },
        {
            "hospital_id": "hosp_003",
            "name": "Yashoda Hospitals",
            "address": "Somajiguda, Hyderabad",
            "distance_km": 5.8,
            "rating": 4.5,
            "doctors": [
                {"name": "Dr. Venkat Rao", "experience": 18, "rating": 4.6, "fee": 750},
            ]
        }
    ],
    "dermatologist": [
        {
            "hospital_id": "hosp_004",
            "name": "Kaya Skin Clinic",
            "address": "Madhapur, Hyderabad",
            "distance_km": 3.1,
            "rating": 4.7,
            "doctors": [
                {"name": "Dr. Meera Nair", "experience": 10, "rating": 4.8, "fee": 500},
                {"name": "Dr. Arun Patel", "experience": 7, "rating": 4.5, "fee": 400},
            ]
        },
        {
            "hospital_id": "hosp_005",
            "name": "Oliva Skin & Hair Clinic",
            "address": "Gachibowli, Hyderabad",
            "distance_km": 6.0,
            "rating": 4.4,
            "doctors": [
                {"name": "Dr. Sneha Gupta", "experience": 12, "rating": 4.6, "fee": 600},
            ]
        }
    ],
    "orthopedic": [
        {
            "hospital_id": "hosp_006",
            "name": "Continental Hospitals",
            "address": "Gachibowli, Hyderabad",
            "distance_km": 5.5,
            "rating": 4.7,
            "doctors": [
                {"name": "Dr. Ramesh Babu", "experience": 22, "rating": 4.9, "fee": 900},
                {"name": "Dr. Kavitha Reddy", "experience": 14, "rating": 4.6, "fee": 700},
            ]
        },
        {
            "hospital_id": "hosp_007",
            "name": "KIMS Hospital",
            "address": "Secunderabad, Hyderabad",
            "distance_km": 8.2,
            "rating": 4.5,
            "doctors": [
                {"name": "Dr. Srinivas Rao", "experience": 16, "rating": 4.7, "fee": 800},
            ]
        }
    ],
    "neurologist": [
        {
            "hospital_id": "hosp_008",
            "name": "NIMS Hospital",
            "address": "Punjagutta, Hyderabad",
            "distance_km": 4.0,
            "rating": 4.8,
            "doctors": [
                {"name": "Dr. Lakshmi Prasad", "experience": 25, "rating": 4.9, "fee": 1200},
                {"name": "Dr. Mohan Krishna", "experience": 15, "rating": 4.7, "fee": 800},
            ]
        }
    ],
    "gastroenterologist": [
        {
            "hospital_id": "hosp_009",
            "name": "Asian Institute of Gastroenterology",
            "address": "Somajiguda, Hyderabad",
            "distance_km": 5.0,
            "rating": 4.9,
            "doctors": [
                {"name": "Dr. Nageshwar Reddy", "experience": 30, "rating": 5.0, "fee": 1500},
                {"name": "Dr. Manu Tandan", "experience": 18, "rating": 4.8, "fee": 1000},
            ]
        }
    ],
    "pulmonologist": [
        {
            "hospital_id": "hosp_010",
            "name": "Chest Hospital",
            "address": "Erragadda, Hyderabad",
            "distance_km": 7.5,
            "rating": 4.4,
            "doctors": [
                {"name": "Dr. Ravi Shankar", "experience": 20, "rating": 4.6, "fee": 600},
                {"name": "Dr. Sunitha Rani", "experience": 12, "rating": 4.5, "fee": 500},
            ]
        }
    ],
    "ophthalmologist": [
        {
            "hospital_id": "hosp_011",
            "name": "LV Prasad Eye Institute",
            "address": "Banjara Hills, Hyderabad",
            "distance_km": 4.5,
            "rating": 4.9,
            "doctors": [
                {"name": "Dr. Gullapalli Rao", "experience": 28, "rating": 4.9, "fee": 800},
                {"name": "Dr. Prashant Garg", "experience": 20, "rating": 4.8, "fee": 700},
            ]
        }
    ],
    "ent_specialist": [
        {
            "hospital_id": "hosp_012",
            "name": "Yashoda ENT Hospital",
            "address": "Malakpet, Hyderabad",
            "distance_km": 6.8,
            "rating": 4.5,
            "doctors": [
                {"name": "Dr. Sanjay Kumar", "experience": 15, "rating": 4.6, "fee": 500},
                {"name": "Dr. Rekha Sharma", "experience": 10, "rating": 4.4, "fee": 400},
            ]
        }
    ],
    "psychiatrist": [
        {
            "hospital_id": "hosp_013",
            "name": "Institute of Mental Health",
            "address": "Erragadda, Hyderabad",
            "distance_km": 7.0,
            "rating": 4.3,
            "doctors": [
                {"name": "Dr. Vijay Kumar", "experience": 18, "rating": 4.5, "fee": 700},
                {"name": "Dr. Padma Rao", "experience": 22, "rating": 4.7, "fee": 900},
            ]
        }
    ],
    "general_physician": [
        {
            "hospital_id": "hosp_014",
            "name": "Apollo Clinic",
            "address": "Kukatpally, Hyderabad",
            "distance_km": 3.0,
            "rating": 4.6,
            "doctors": [
                {"name": "Dr. Ramana Murthy", "experience": 20, "rating": 4.7, "fee": 400},
                {"name": "Dr. Swathi Reddy", "experience": 8, "rating": 4.5, "fee": 300},
            ]
        },
        {
            "hospital_id": "hosp_015",
            "name": "Max Healthcare",
            "address": "Madhapur, Hyderabad",
            "distance_km": 2.8,
            "rating": 4.5,
            "doctors": [
                {"name": "Dr. Kiran Kumar", "experience": 12, "rating": 4.6, "fee": 350},
            ]
        }
    ]
}


def get_hospitals_by_specialist(specialist_type: str) -> List[Hospital]:
    """
    Fetch hospitals and doctors for a given specialist type.
    
    Chain of Thought:
    - Look up specialist type in mock data
    - Generate fresh time slots for each doctor
    - Return structured Hospital objects with nested Doctor objects
    - If specialist not found, return general physician data
    """
    specialist_key = specialist_type.lower().replace(" ", "_")
    
    if specialist_key not in MOCK_HOSPITALS_DATA:
        specialist_key = "general_physician"
    
    hospitals_data = MOCK_HOSPITALS_DATA[specialist_key]
    hospitals = []
    
    for hosp_data in hospitals_data:
        doctors = []
        for doc_data in hosp_data["doctors"]:
            doctor_id = f"doc_{uuid.uuid4().hex[:8]}"
            doctors.append(Doctor(
                doctor_id=doctor_id,
                name=doc_data["name"],
                specialization=specialist_type.replace("_", " ").title(),
                experience_years=doc_data["experience"],
                rating=doc_data["rating"],
                consultation_fee=doc_data["fee"],
                hospital_id=hosp_data["hospital_id"],
                available_slots=generate_time_slots(doctor_id)
            ))
        
        hospitals.append(Hospital(
            hospital_id=hosp_data["hospital_id"],
            name=hosp_data["name"],
            address=hosp_data["address"],
            distance_km=hosp_data["distance_km"],
            rating=hosp_data["rating"],
            doctors=doctors
        ))
    
    return hospitals
