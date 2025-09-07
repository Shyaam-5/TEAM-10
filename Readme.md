# MedRef - AI-Powered Medical Referral Platform

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)](https://flask.palletsprojects.com/)

A comprehensive Flask application that leverages machine learning and AI to optimize medical specialist referrals, reduce wait times, and improve patient outcomes.

## 🚀 Features

### Core Functionality
- **AI-Powered Specialist Recommendations**: Uses machine learning models to analyze patient notes and recommend appropriate specialists
- **LLM Integration**: Gemini AI for natural language processing and medical data extraction
- **Dual Validation System**: ML predictions validated by LLM for improved accuracy
- **Smart Clinic Ranking**: Dynamic ranking based on wait times, distance, and consultation costs
- **Real-time Maps Integration**: Google Places API and Distance Matrix for clinic locations

### User Management
- **Multi-Role Authentication**: Support for clinicians and patients
- **Multiple Login Methods**: Traditional email/password, Google OAuth
- **Database Flexibility**: Snowflake integration with local JSON fallback
- **Session Management**: Secure JWT-based authentication

### Advanced Features
- **Dr. Ellie Chatbot**: RAG-powered assistant using Pinecone vector database
- **Voice Transcription**: Groq API with Whisper for voice-to-text
- **Wait Time Prediction**: ML models for predicting specialist availability
- **Patient Dashboard**: Referral tracking and follow-up system
- **Dynamic API**: RESTful endpoints for integration

## 🛠️ Technology Stack

- **Backend**: Flask (Python)
- **Machine Learning**: scikit-learn, joblib
- **AI/LLM**: Google Gemini, LangChain, Pinecone
- **Database**: Snowflake (primary), JSON files (fallback)
- **APIs**: Google Places, Google Maps, Groq
- **Authentication**: Flask-JWT-Extended, Authlib
- **Frontend**: HTML templates with JavaScript

## 📋 Prerequisites

- Python 3.8+
- Required model files:
  - `advanced_ensemble_pipeline.pkl`
  - `referral_label_encoder.pkl`
  - `wait_time_predictor.pkl` (optional)

## ⚙️ Installation

1. **Clone the repository**

2. **Create virtual environment**

3. **Set up environment variables**

Create a `.env` file in the root directory:
Required API Keys

GEMINI_API_KEY=your_gemini_api_key

GOOGLE_PLACES_API_KEY=your_google_places_key

GROQ_API_KEY=your_groq_api_key

PINECONE_API_KEY=your_pinecone_key

Authentication

SECRET_KEY=your_secret_key

JWT_SECRET_KEY=your_jwt_secret_key

GOOGLE_CLIENT_ID=your_google_oauth_client_id

GOOGLE_CLIENT_SECRET=your_google_oauth_client_secret

Snowflake (Optional - will use local files if not provided)

SNOWFLAKE_USER=your_snowflake_user

SNOWFLAKE_PASSWORD=your_snowflake_password

SNOWFLAKE_ACCOUNT=your_snowflake_account

SNOWFLAKE_WAREHOUSE=your_warehouse

SNOWFLAKE_DATABASE=your_database

SNOWFLAKE_SCHEMA=your_schema

SNOWFLAKE_ROLE=your_role

Model Paths (Optional - uses defaults if not specified)

MODEL_PATH=advanced_ensemble_pipeline.pkl

ENCODER_PATH=referral_label_encoder.pkl

WAIT_TIME_MODEL_PATH=wait_time_predictor.pkl



5. **Initialize Pinecone Index**

Ensure you have a Pinecone index named "ellie" with appropriate medical documents indexed.

## 🚀 Usage

### Starting the Application



The application will be available at `http://localhost:5000`

### Key Endpoints

#### Web Interface
- `/` - Landing page
- `/signup` - Clinician registration
- `/patient/signup` - Patient registration  
- `/login` - Clinician login
- `/patient/login` - Patient login
- `/referral` - Main referral analysis tool
- `/ranking` - Clinic rankings and maps
- `/patient/dashboard` - Patient referral history

#### API Endpoints
- `POST /api/referral` - Generate specialist recommendations
- `POST /api/dynamic-referral` - Advanced clinic ranking with wait times
- `POST /api/chatbot/chat` - Chat with Dr. Ellie
- `POST /api/voice/transcribe` - Voice-to-text transcription
- `GET /api/patient/dashboard_data` - Patient referral data

### Workflow Example

1. **Clinician Login**: Access the referral tool
2. **Patient Analysis**: Enter patient notes and email
3. **AI Processing**: System extracts features and predicts specialist
4. **LLM Validation**: Gemini AI validates and potentially overrides prediction
5. **Clinic Ranking**: View ranked specialists by wait time, distance, and cost
6. **Patient Notification**: Patient receives referral information via dashboard

## 🏗️ Architecture

### ML Pipeline
1. **Data Extraction**: LLM extracts structured data from clinical notes
2. **Feature Engineering**: Creates derived features for ML model
3. **Prediction**: Ensemble model predicts specialist recommendation
4. **Validation**: LLM reviews and potentially overrides ML prediction
5. **Explanation**: Generates human-readable explanations

### Database Schema
The application supports both Snowflake and local JSON storage:
- `users` - User authentication and profiles
- `referrals` - Referral records with predictions and explanations
- `specialists` - Healthcare provider information
- `appointments` - Appointment scheduling
- `patient_followups` - Post-referral tracking

## 🔧 Configuration

### Fallback Behavior
The application is designed to degrade gracefully:
- **No Snowflake**: Uses local JSON files
- **No API Keys**: Provides fallback responses
- **Missing Models**: Returns default recommendations

### Supported Specialists
- Cardiology
- Dermatology  
- Endocrinology
- Gastroenterology
- General/Internal Medicine
- Hematology
- Pulmonology
- Infectious Diseases
- Nephrology/Urology
- Neurology
- Oncology
- Primary Care Management
- Psychiatry
- Rheumatology/Orthopedics

## 📊 Features in Detail

### AI-Powered Analysis
- Processes natural language patient notes
- Extracts 20+ clinical features automatically
- Provides confidence scores for recommendations
- Supports medical relevance validation

### Smart Clinic Ranking  
- Real-time wait time predictions
- Distance-based ranking with driving directions
- Cost comparison across providers
- Interactive maps with clinic locations

### Dr. Ellie Chatbot
- RAG-powered medical assistant
- Pinecone vector database integration
- Conversational memory for context
- Fallback responses when offline

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

For questions or issues:
1. Check the application logs for detailed error messages
2. Ensure all required model files are present
3. Verify API keys are correctly configured
4. Review the fallback behavior documentation

The application includes comprehensive error handling and will provide helpful error messages to guide troubleshooting.

## 👥 Contributors

- Your Name - Initial work - [YourGitHub](https://github.com/yourusername)

## 📝 Changelog

See [CHANGELOG.md](CHANGELOG.md) for a list of changes and version history.

## 🏷️ Tags

`healthcare` `machine-learning` `ai` `flask` `medical-referral` `nlp` `gemini` `python`



1. **Clone the repository**
