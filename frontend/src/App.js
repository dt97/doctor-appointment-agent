import React, { useState, useEffect, useRef } from 'react';
import { Send, Bot, User, Calendar, MapPin, Star, Clock, CheckCircle, AlertCircle, Stethoscope } from 'lucide-react';
import axios from 'axios';
import './App.css';

const API_BASE_URL = 'http://localhost:8000/api';

function App() {
  const [messages, setMessages] = useState([]);
  const [inputMessage, setInputMessage] = useState('');
  const [sessionId, setSessionId] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [currentState, setCurrentState] = useState('symptom_collection');
  const [doctorData, setDoctorData] = useState(null);
  const [selectedDoctor, setSelectedDoctor] = useState(null);
  const [selectedHospital, setSelectedHospital] = useState(null);
  const [selectedSlot, setSelectedSlot] = useState(null);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    initSession();
  }, []);

  const initSession = async () => {
    try {
      const response = await axios.post(`${API_BASE_URL}/session`, {});
      setSessionId(response.data.session_id);
      setMessages([{
        role: 'assistant',
        content: response.data.message,
        type: response.data.message_type
      }]);
    } catch (error) {
      console.error('Failed to initialize session:', error);
      setMessages([{
        role: 'assistant',
        content: 'Hello! I\'m your medical appointment assistant. Please tell me about your medical concern or symptoms.',
        type: 'text'
      }]);
      setSessionId('local-' + Date.now());
    }
  };

  const sendMessage = async (message, selectedData = null) => {
    if (!message.trim() && !selectedData) return;

    const userMessage = message || 'Selected appointment slot';
    setMessages(prev => [...prev, { role: 'user', content: userMessage, type: 'text' }]);
    setInputMessage('');
    setIsLoading(true);

    try {
      const response = await axios.post(`${API_BASE_URL}/chat`, {
        message: userMessage,
        session_id: sessionId,
        selected_data: selectedData
      });

      const assistantMessage = {
        role: 'assistant',
        content: response.data.message,
        type: response.data.message_type,
        data: response.data.data
      };

      setMessages(prev => [...prev, assistantMessage]);
      setCurrentState(response.data.state);

      if (response.data.message_type === 'doctor_selection' && response.data.data?.hospitals) {
        setDoctorData(response.data.data);
      }

      if (response.data.state === 'booking_confirmation' || response.data.state === 'completed') {
        setDoctorData(null);
        setSelectedDoctor(null);
        setSelectedHospital(null);
        setSelectedSlot(null);
      }
    } catch (error) {
      console.error('Failed to send message:', error);
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: 'Sorry, I encountered an error. Please try again.',
        type: 'error'
      }]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage(inputMessage);
    }
  };

  const handleSlotSelection = () => {
    if (selectedDoctor && selectedHospital && selectedSlot) {
      sendMessage('I have selected my appointment', {
        doctor_id: selectedDoctor.doctor_id,
        hospital_id: selectedHospital.hospital_id,
        slot_id: selectedSlot.slot_id
      });
    }
  };

  const renderMessage = (msg, index) => {
    const isUser = msg.role === 'user';

    return (
      <div key={index} className={`message-wrapper ${isUser ? 'user' : 'assistant'}`}>
        <div className={`message-avatar ${isUser ? 'user-avatar' : 'bot-avatar'}`}>
          {isUser ? <User size={20} /> : <Bot size={20} />}
        </div>
        <div className={`message-bubble ${isUser ? 'user-bubble' : 'assistant-bubble'}`}>
          <div className="message-content">
            {msg.content.split('\n').map((line, i) => (
              <p key={i}>{line}</p>
            ))}
          </div>
        </div>
      </div>
    );
  };

  const renderDoctorSelection = () => {
    if (!doctorData?.hospitals) return null;

    return (
      <div className="doctor-selection-panel">
        <div className="panel-header">
          <Stethoscope size={24} />
          <h2>Available {doctorData.specialist_type}s</h2>
        </div>

        <div className="hospitals-list">
          {doctorData.hospitals.map((hospital) => (
            <div key={hospital.hospital_id} className="hospital-card">
              <div className="hospital-header">
                <div className="hospital-info">
                  <h3>{hospital.name}</h3>
                  <p className="hospital-address">
                    <MapPin size={14} /> {hospital.address}
                  </p>
                  <div className="hospital-meta">
                    <span className="rating"><Star size={14} /> {hospital.rating}</span>
                    <span className="distance">{hospital.distance_km} km away</span>
                  </div>
                </div>
              </div>

              <div className="doctors-list">
                {hospital.doctors.map((doctor) => (
                  <div
                    key={doctor.doctor_id}
                    className={`doctor-card ${selectedDoctor?.doctor_id === doctor.doctor_id ? 'selected' : ''}`}
                    onClick={() => {
                      setSelectedDoctor(doctor);
                      setSelectedHospital(hospital);
                      setSelectedSlot(null);
                    }}
                  >
                    <div className="doctor-info">
                      <h4>{doctor.name}</h4>
                      <p className="specialization">{doctor.specialization}</p>
                      <div className="doctor-meta">
                        <span>{doctor.experience_years} yrs exp</span>
                        <span><Star size={12} /> {doctor.rating}</span>
                        <span className="fee">₹{doctor.consultation_fee}</span>
                      </div>
                    </div>

                    {selectedDoctor?.doctor_id === doctor.doctor_id && (
                      <div className="slots-container">
                        <p className="slots-label"><Clock size={14} /> Available Slots:</p>
                        <div className="slots-grid">
                          {doctor.available_slots
                            .filter(slot => slot.available)
                            .slice(0, 12)
                            .map((slot) => (
                              <button
                                key={slot.slot_id}
                                className={`slot-btn ${selectedSlot?.slot_id === slot.slot_id ? 'selected' : ''}`}
                                onClick={(e) => {
                                  e.stopPropagation();
                                  setSelectedSlot(slot);
                                }}
                              >
                                <span className="slot-date">{slot.date}</span>
                                <span className="slot-time">{slot.time}</span>
                              </button>
                            ))}
                        </div>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>

        {selectedDoctor && selectedSlot && (
          <div className="selection-summary">
            <div className="summary-content">
              <CheckCircle size={20} />
              <span>
                <strong>{selectedDoctor.name}</strong> at <strong>{selectedHospital?.name}</strong>
                <br />
                {selectedSlot.date} at {selectedSlot.time}
              </span>
            </div>
            <button className="confirm-selection-btn" onClick={handleSlotSelection}>
              Confirm Selection
            </button>
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="app-container">
      <header className="app-header">
        <div className="header-content">
          <div className="logo">
            <Stethoscope size={32} />
            <span>MedBook AI</span>
          </div>
          <p className="tagline">Your AI-Powered Medical Appointment Assistant</p>
        </div>
      </header>

      <main className="main-content">
        <div className={`chat-container ${doctorData ? 'with-panel' : ''}`}>
          <div className="chat-panel">
            <div className="chat-header">
              <Bot size={24} />
              <span>Medical Assistant</span>
              <div className={`status-indicator ${currentState}`}>
                {currentState.replace(/_/g, ' ')}
              </div>
            </div>

            <div className="messages-container">
              {messages.map((msg, index) => renderMessage(msg, index))}
              {isLoading && (
                <div className="message-wrapper assistant">
                  <div className="message-avatar bot-avatar">
                    <Bot size={20} />
                  </div>
                  <div className="message-bubble assistant-bubble">
                    <div className="typing-indicator">
                      <span></span>
                      <span></span>
                      <span></span>
                    </div>
                  </div>
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>

            <div className="input-container">
              <input
                type="text"
                value={inputMessage}
                onChange={(e) => setInputMessage(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder="Type your message..."
                disabled={isLoading}
              />
              <button
                onClick={() => sendMessage(inputMessage)}
                disabled={isLoading || !inputMessage.trim()}
                className="send-btn"
              >
                <Send size={20} />
              </button>
            </div>
          </div>

          {doctorData && (
            <div className="selection-panel">
              {renderDoctorSelection()}
            </div>
          )}
        </div>
      </main>

      <footer className="app-footer">
        <p>Powered by LangChain AI • For demonstration purposes only</p>
      </footer>
    </div>
  );
}

export default App;
