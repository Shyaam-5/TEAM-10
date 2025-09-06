# app.py
import os
import re
import json
import joblib
import numpy as np
import pandas as pd
import requests
import snowflake.connector
import random
from dotenv import load_dotenv
from flask import (
    Flask, render_template, request, redirect, url_for, session, flash, jsonify
)
from werkzeug.security import generate_password_hash, check_password_hash
from flask_jwt_extended import (
    create_access_token, jwt_required, JWTManager, get_jwt_identity
)
from authlib.integrations.flask_client import OAuth

# LangChain / Gemini
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import Pinecone
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from pinecone import Pinecone as PineconeClient
from groq import Groq
import base64
import io
from datetime import timedelta, datetime, date, time
import math
import threading

# ---------------------------
# Load environment
# ---------------------------
load_dotenv()

# Required env vars
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GOOGLE_PLACES_API_KEY = os.getenv("GOOGLE_PLACES_API_KEY")
SNOWFLAKE_REQUIRED = all([
    os.getenv("SNOWFLAKE_USER"),
    os.getenv("SNOWFLAKE_PASSWORD"),
    os.getenv("SNOWFLAKE_ACCOUNT"),
    os.getenv("SNOWFLAKE_WAREHOUSE"),
    os.getenv("SNOWFLAKE_DATABASE"),
    os.getenv("SNOWFLAKE_SCHEMA"),
    os.getenv("SNOWFLAKE_ROLE"),
])

# Allow running without API keys for local/dev; features that need them will fallback
if not GEMINI_API_KEY:
    print("Warning: GEMINI_API_KEY missing in .env — LLM features will use fallbacks.")
if not GOOGLE_PLACES_API_KEY:
    print("Warning: GOOGLE_PLACES_API_KEY missing in .env — Maps features will be limited.")
if not SNOWFLAKE_REQUIRED:
    # Not fatal — allow running without Snowflake if you only want to test ML parts.
    print("Warning: Snowflake env variables not fully set. Signup/login using Snowflake will fail.")

# ---------------------------
# Local users fallback (when Snowflake is unavailable)
# ---------------------------
LOCAL_USERS_FILE = os.getenv("LOCAL_USERS_FILE", os.path.join(os.path.dirname(__file__), "local_users.json"))
LOCAL_ROLES_FILE = os.getenv("LOCAL_ROLES_FILE", os.path.join(os.path.dirname(__file__), "local_roles.json"))
_users_file_lock = threading.Lock()
_roles_file_lock = threading.Lock()

def _load_local_users() -> list:
    try:
        if not os.path.exists(LOCAL_USERS_FILE):
            return []
        with _users_file_lock:
            with open(LOCAL_USERS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
                return []
    except Exception:
        return []

def _save_local_users(users: list) -> None:
    with _users_file_lock:
        with open(LOCAL_USERS_FILE, "w", encoding="utf-8") as f:
            json.dump(users, f, ensure_ascii=False, indent=2)

def _next_local_user_id(users: list) -> int:
    if not users:
        return 1
    try:
        return max(int(u.get("id", 0)) for u in users) + 1
    except Exception:
        return len(users) + 1

def _load_local_roles() -> dict:
    try:
        if not os.path.exists(LOCAL_ROLES_FILE):
            return {}
        with _roles_file_lock:
            with open(LOCAL_ROLES_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, dict) else {}
    except Exception:
        return {}

def _save_local_roles(mapping: dict) -> None:
    with _roles_file_lock:
        with open(LOCAL_ROLES_FILE, "w", encoding="utf-8") as f:
            json.dump(mapping, f, ensure_ascii=False, indent=2)

# ---------------------------
# Flask app + JWT + OAuth
# ---------------------------
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "dev-secret")
app.config["JWT_SECRET_KEY"] = os.getenv("JWT_SECRET_KEY", "jwt-secret")
app.config.update(
    SECRET_KEY=os.getenv("SECRET_KEY", "your-super-secret-key-here"),
    SESSION_COOKIE_SECURE=False,  # Set to True in production with HTTPS
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE='Lax',
    PERMANENT_SESSION_LIFETIME=timedelta(hours=24)
)

# Make sessions permanent by default
@app.before_request
def make_session_permanent():
    session.permanent = True
jwt = JWTManager(app)

oauth = OAuth(app)
google = oauth.register(
    name='google',
    client_id=os.getenv("GOOGLE_CLIENT_ID"),
    client_secret=os.getenv("GOOGLE_CLIENT_SECRET"),
    access_token_url='https://accounts.google.com/o/oauth2/token',
    authorize_url='https://accounts.google.com/o/oauth2/auth',
    api_base_url='https://www.googleapis.com/oauth2/v1/',
    userinfo_endpoint='https://openidconnect.googleapis.com/v1/userinfo',
    client_kwargs={'scope': 'email profile'},
)

# ---------------------------
# Snowflake helper
# ---------------------------
def get_connection():
    if not SNOWFLAKE_REQUIRED:
        raise RuntimeError("Snowflake configuration is missing")
    return snowflake.connector.connect(
        user=os.getenv("SNOWFLAKE_USER"),
        password=os.getenv("SNOWFLAKE_PASSWORD"),
        account=os.getenv("SNOWFLAKE_ACCOUNT"),
        warehouse=os.getenv("SNOWFLAKE_WAREHOUSE"),
        database=os.getenv("SNOWFLAKE_DATABASE"),
        schema=os.getenv("SNOWFLAKE_SCHEMA"),
        role=os.getenv("SNOWFLAKE_ROLE")
    )

def _fq_table(table_name: str) -> str:
    """Return fully qualified table name for Snowflake queries."""
    db = os.getenv("SNOWFLAKE_DATABASE")
    schema = os.getenv("SNOWFLAKE_SCHEMA")
    if db and schema:
        return f"{db}.{schema}.{table_name}"
    return table_name

def _log_db_error(err: Exception, context: str):
    try:
        err_text = str(err)
    except Exception:
        err_text = "<unstringifiable error>"
    print(f"[DB ERROR] Context={context} Error={err_text}")

def _diagnose_connection():
    diag = {
        "SNOWFLAKE_REQUIRED": bool(SNOWFLAKE_REQUIRED),
        "database": os.getenv("SNOWFLAKE_DATABASE"),
        "schema": os.getenv("SNOWFLAKE_SCHEMA"),
        "warehouse": os.getenv("SNOWFLAKE_WAREHOUSE"),
        "role": os.getenv("SNOWFLAKE_ROLE"),
        "connected": False,
        "current_version": None,
        "referrals_exists": None,
        "referrals_count": None,
        "errors": []
    }
    if not SNOWFLAKE_REQUIRED:
        diag["errors"].append("Snowflake env not fully configured")
        return diag
    try:
        conn = get_connection()
        cur = conn.cursor()
        diag["connected"] = True
        try:
            cur.execute("SELECT CURRENT_VERSION()")
            diag["current_version"] = cur.fetchone()[0]
        except Exception as e:
            _log_db_error(e, "SELECT CURRENT_VERSION()")
            diag["errors"].append(f"version: {e}")
        try:
            cur.execute(f"SHOW TABLES LIKE 'REFERRALS' IN SCHEMA {os.getenv('SNOWFLAKE_DATABASE')}.{os.getenv('SNOWFLAKE_SCHEMA')}")
            diag["referrals_exists"] = cur.fetchone() is not None
        except Exception as e:
            _log_db_error(e, "SHOW TABLES referrals")
            diag["errors"].append(f"show tables: {e}")
        try:
            cur.execute(f"SELECT COUNT(*) FROM {_fq_table('referrals')}")
            diag["referrals_count"] = cur.fetchone()[0]
        except Exception as e:
            _log_db_error(e, "COUNT referrals")
            diag["errors"].append(f"count referrals: {e}")
        try:
            cur.close(); conn.close()
        except Exception:
            pass
    except Exception as e:
        _log_db_error(e, "connect")
        diag["errors"].append(f"connect: {e}")
    return diag

# Utility: safe DF from cursor
def df_from_cursor(cur) -> pd.DataFrame:
    rows = cur.fetchall()
    cols = [d[0] for d in cur.description] if cur.description else []
    return pd.DataFrame(rows, columns=cols)

# ---------------------------
# Load Model + Label Encoder
# ---------------------------
MODEL_PATH = os.getenv("MODEL_PATH", "advanced_ensemble_pipeline.pkl")
ENCODER_PATH = os.getenv("ENCODER_PATH", "referral_label_encoder.pkl")
WAIT_TIME_MODEL_PATH = os.getenv("WAIT_TIME_MODEL_PATH", "wait_time_predictor.pkl")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
if not os.path.exists(ENCODER_PATH):
    raise FileNotFoundError(f"Encoder file not found: {ENCODER_PATH}")

model_pipeline = joblib.load(MODEL_PATH)
label_encoder = joblib.load(ENCODER_PATH)
wait_time_model = None
if os.path.exists(WAIT_TIME_MODEL_PATH):
    try:
        wait_time_model = joblib.load(WAIT_TIME_MODEL_PATH)
    except Exception as _e:
        print(f"Warning: failed to load wait time model: {_e}")

# Chatbot variables
qa_chain = None

# ---------------------------
# Chatbot setup
# ---------------------------
def build_chatbot_pipeline():
    """Build the RAG pipeline for Dr. Ellie chatbot using Pinecone"""
    try:
        print("[*] Building Dr. Ellie RAG pipeline with Pinecone...")
        
        # Get API keys from environment
        google_api_key = os.getenv("GEMINI_API_KEY")
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        
        if not google_api_key:
            print("Warning: GEMINI_API_KEY not found in environment variables - using fallback responses")
            return None
            
        if not pinecone_api_key:
            print("Warning: PINECONE_API_KEY not found in environment variables - using fallback responses")
            return None
            
        # Initialize Pinecone client
        pc = PineconeClient(api_key=pinecone_api_key)
        index_name = "ellie"  # Your existing index
        
        # Check if index exists
        if index_name not in pc.list_indexes().names():
            print(f"Warning: Pinecone index '{index_name}' not found - using fallback responses")
            return None
            
        print(f"[*] Connecting to existing Pinecone index: {index_name}")
        
        # Initialize embeddings with Google API key
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001", 
            google_api_key=google_api_key
        )
        
        # Connect to existing Pinecone index
        vectordb = Pinecone.from_existing_index(index_name, embeddings)

        # Initialize LLM with explicit API key
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash", 
            temperature=0, 
            google_api_key=google_api_key
        )
        retriever = vectordb.as_retriever(search_kwargs={"k": 2})

        # Memory for chat continuity
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

        # Custom prompt for Dr. Ellie
        custom_prompt_template = """
You are Dr. Ellie, a friendly chatbot that assists doctors in using the MedRef website.
Think of yourself as a supportive friend guiding them, not a manual or a textbook.

Rules for how you speak:
- Only use the context provided below to answer. Do not make up or guess answers.
- Answer in a warm, natural way like real conversation.
- Do NOT use headings, bullet points, or numbered lists.
- Keep explanations simple, flowing, and easy to follow.
- Use connecting phrases like "first you'll want to...", "then you can...", "after that..." instead of lists.
- Ask light follow-up questions to keep the conversation human-like.
- If information is missing, say so politely but continue guiding.
- do not mention about the documents in your response.
- also keep your response short and concise.

Context from documents:
{context}

User question:
{question}

Your reply as Dr. Ellie:
"""
        prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])

        # Conversational RAG chain
        qa = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            combine_docs_chain_kwargs={"prompt": prompt}
        )
        return qa
    except Exception as e:
        print(f"Error building chatbot pipeline: {e}")
        return None

# ---------------------------
# LLM prompts
# ---------------------------
# Extraction prompt — exact schema, includes Location
extract_prompt = PromptTemplate(
    input_variables=["note"],
    template="""
You are a careful medical data extraction assistant.
Extract ONLY the following fields from the patient note. The JSON keys and data types must EXACTLY match.

Rules:
- If a field is missing in the note, provide the default indicated.
- "Gender": "Male" | "Female" | "Unknown"
- All boolean features: 0 or 1
- Age_Group: "Child","Teen","Young_Adult","Adult","Senior","Elderly","Unknown"
- Duration_Group: "Acute","Subacute","Chronic","Longterm","Very_Longterm","Unknown"
- Location: a recognizable city/town (e.g., 'New York', 'Mysore', 'Chennai') or "Unknown"
- DO NOT return null values.

Patient Note:
{note}

Return ONLY valid JSON (no markdown/comments) with EXACT keys and types:

{{
  "Age": <int or 0>,
  "Gender": <"Male"|"Female"|"Unknown">,
  "Symptom_Duration_Days": <int or 0>,
  "Symptoms": "<string or ''>",
  "Past_History": "<string or ''>",
  "Location": "<string or 'Unknown'>",
  "has_chest_pain": <0|1>,
  "has_headache": <0|1>,
  "has_neurological": <0|1>,
  "has_respiratory": <0|1>,
  "has_gastrointestinal": <0|1>,
  "has_fever": <0|1>,
  "has_bleeding": <0|1>,
  "has_swelling": <0|1>,
  "chest_pain_intensity": <int or 0>,
  "headache_intensity": <int or 0>,
  "neurological_intensity": <int or 0>,
  "respiratory_intensity": <int or 0>,
  "gastrointestinal_intensity": <int or 0>,
  "fever_intensity": <int or 0>,
  "bleeding_intensity": <int or 0>,
  "swelling_intensity": <int or 0>,
  "Age_Group": <"Child"|"Teen"|"Young_Adult"|"Adult"|"Senior"|"Elderly"|"Unknown">,
  "Duration_Group": <"Acute"|"Subacute"|"Chronic"|"Longterm"|"Very_Longterm"|"Unknown">
}}
"""
)
relevance_prompt = PromptTemplate(
    input_variables=["note"],
    template="""
    Analyze the following patient note to determine if it is medically relevant for a specialist referral.
    A note is RELEVANT if it describes patient symptoms, medical history, or a health condition.
    A note is IRRELEVANT if it is nonsense, random text, a generic question, or clearly not a patient note.

    Patient Note:
    "{note}"

    Based on the criteria, is this note Relevant or Irrelevant?
    Respond with ONLY the single word: 'Relevant' or 'Irrelevant'.
    """
)
validation_prompt = PromptTemplate(
    input_variables=["note", "ml_specialist", "specialist_list"],
    template="""
    You are an expert medical diagnostician reviewing a prediction from an AI model.
    Your task is to validate the model's recommendation and correct it if necessary from the provided list.

    Possible Specialists: {specialist_list}

    Patient Note:
    "{note}"

    The initial AI model recommended: {ml_specialist}

    Review the patient note. Is "{ml_specialist}" the most appropriate specialist from the list?
    If not, provide the single most appropriate specialist.

    Return a single, valid JSON object with two keys:
    1. "validated_specialist": The most appropriate specialist from the list. Must be one of {specialist_list}.
    2. "validation_reason": A brief, 1-2 sentence justification for your choice.

    Return ONLY the valid JSON object.
    """)
def parse_validation_output(raw_output: str) -> dict:
    """
    Safely parses the JSON output from the validation LLM.
    """
    try:
        # The LLM might wrap the JSON in markdown backticks
        match = re.search(r'\{.*\}', raw_output, re.DOTALL)
        if match:
            return json.loads(match.group())
        return json.loads(raw_output)
    except (json.JSONDecodeError, AttributeError):
        # Handle cases where parsing fails
        print(f"[WARN] Failed to parse validation JSON: {raw_output}")
        return {
            "validated_specialist": None,
            "validation_reason": "Could not determine validation result."
        }

# Explainability prompt
explain_prompt = PromptTemplate(
    input_variables=["note", "specialist", "top_features"],
    template="""
You are an explainable medical assistant. Given the patient note:
{note}

The model recommended: {specialist}

Top features (name: value):
{top_features}

Explain in 2-4 short sentences why the referral to {specialist} makes clinical sense, referencing key symptoms or features.
"""
)
SPECIALIST_LIST = [
    'Cardiology','Dermatology','Endocrinology',
    'Gastroenterology','General/Internal Medicine','Hematology','Pulmonology','Infectious Diseases',
    'Nephrology/Urology','Neurology','Oncology','Primary Care Management','Psychiatry','Rheumatology/Orthopedics'
]
# LLM chain setup with safe fallbacks when no GEMINI_API_KEY
class _FallbackChain:
    def __init__(self, kind: str):
        self.kind = kind
    def run(self, vars_dict):
        if self.kind == 'extract':
            # Minimal valid JSON matching expected schema
            return json.dumps({
                "Age": 0,
                "Gender": "Unknown",
                "Symptom_Duration_Days": 0,
                "Symptoms": "",
                "Past_History": "",
                "Location": "Unknown",
                "has_chest_pain": 0,
                "has_headache": 0,
                "has_neurological": 0,
                "has_respiratory": 0,
                "has_gastrointestinal": 0,
                "has_fever": 0,
                "has_bleeding": 0,
                "has_swelling": 0,
                "chest_pain_intensity": 0,
                "headache_intensity": 0,
                "neurological_intensity": 0,
                "respiratory_intensity": 0,
                "gastrointestinal_intensity": 0,
                "fever_intensity": 0,
                "bleeding_intensity": 0,
                "swelling_intensity": 0,
                "Age_Group": "Unknown",
                "Duration_Group": "Unknown"
            })
        if self.kind == 'explain':
            specialist = vars_dict.get('specialist', 'Specialist')
            return f"Recommended {specialist} based on entered details."
        return ""

if GEMINI_API_KEY:
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0, google_api_key=GEMINI_API_KEY)
        extract_chain = LLMChain(llm=llm, prompt=extract_prompt)
        explain_chain = LLMChain(llm=llm, prompt=explain_prompt)
        relevance_chain = LLMChain(llm=llm, prompt=relevance_prompt)
        validation_chain = LLMChain(llm=llm, prompt=validation_prompt)
    except Exception as _e:
        print(f"Warning: Failed to initialize LLM chains: {_e}. Using fallbacks.")
        extract_chain = _FallbackChain('extract')
        explain_chain = _FallbackChain('explain')
else:
    extract_chain = _FallbackChain('extract')
    explain_chain = _FallbackChain('explain')

# ---------------------------
# Utilities: parse, engineer, prepare input
# ---------------------------
def _strip_codeblocks(text: str) -> str:
    return re.sub(r"```(?:json)?|```", "", text).strip()

def parse_llm_output(raw_result: str):
    cleaned = _strip_codeblocks(raw_result)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse LLM output as JSON: {e}\nRaw:\n{raw_result}")

    # Ensure Location present
    if "Location" not in data or not data["Location"]:
        data["Location"] = "Unknown"

    # Numeric normalization
    int_fields = [
        "Age", "Symptom_Duration_Days",
        "has_chest_pain", "has_headache", "has_neurological", "has_respiratory",
        "has_gastrointestinal", "has_fever", "has_bleeding", "has_swelling",
        "chest_pain_intensity", "headache_intensity", "neurological_intensity",
        "respiratory_intensity", "gastrointestinal_intensity", "fever_intensity",
        "bleeding_intensity", "swelling_intensity"
    ]
    for k in int_fields:
        if k in data:
            try:
                data[k] = int(data[k])
            except (ValueError, TypeError):
                data[k] = 0
        else:
            data[k] = 0

    # Categorical normalization
    for field in ["Gender", "Age_Group", "Duration_Group"]:
        if field not in data or data[field] in [None, "", 0, "0", "null"]:
            data[field] = "Unknown"
        else:
            # Keep as-is (LLM instructed to output exact case)
            data[field] = str(data[field])

    # Text fields
    for field in ["Symptoms", "Past_History"]:
        if field not in data or data[field] is None:
            data[field] = ""
        else:
            data[field] = str(data[field])

    return data
def parse_explanation_output(raw_result: str) -> str:
    """Parse LLM explanation output, return only the explanation text."""
    cleaned = _strip_codeblocks(raw_result)
    try:
        data = json.loads(cleaned)
        # Some models output dict with "text"
        if isinstance(data, dict) and "text" in data:
            return data["text"]
        return cleaned
    except json.JSONDecodeError:
        # If it’s not JSON, just return raw cleaned text
        return cleaned

def engineer_features(structured_data: dict) -> dict:
    # replicate your feature engineering
    d = dict(structured_data)  # shallow copy
    symptoms = d.get('Symptoms', '')
    past_history = d.get('Past_History', '')

    d['Symptom_Count'] = len([s for s in symptoms.split('|') if s.strip()]) if symptoms else 0
    d['Past_History_Count'] = len([s for s in past_history.split('|') if s.strip()]) if past_history else 0
    d['Total_Conditions'] = d['Symptom_Count'] + d['Past_History_Count']

    age = int(d.get('Age', 0))
    d['Age_Squared'] = age ** 2
    d['Age_Log'] = np.log1p(age) if age > 0 else 0

    duration = int(d.get('Symptom_Duration_Days', 0))
    d['Duration_Log'] = np.log1p(duration) if duration > 0 else 0

    d['Age_Symptom_Interaction'] = age * d['Symptom_Count']
    d['Age_Duration_Interaction'] = age * duration

    d['Symptoms_Length'] = len(symptoms)
    d['Past_History_Length'] = len(past_history)
    d['Text_Complexity'] = d['Symptoms_Length'] + d['Past_History_Length']

    d['Combined_Text'] = f"{symptoms} | {past_history}"

    return d

# Keep this list aligned with training order your pipeline expects
REQUIRED_COLUMNS = [
    'Symptom_Duration_Days', 'Age', 'Symptom_Count', 'Past_History_Count',
    'Total_Conditions', 'Age_Squared', 'Age_Log', 'Duration_Log',
    'Age_Symptom_Interaction', 'Age_Duration_Interaction',
    'Symptoms_Length', 'Past_History_Length', 'Text_Complexity',
    'Gender', 'Age_Group', 'Duration_Group',
    'has_chest_pain', 'has_headache', 'has_neurological', 'has_respiratory',
    'has_gastrointestinal', 'has_fever', 'has_bleeding', 'has_swelling',
    'chest_pain_intensity', 'headache_intensity', 'neurological_intensity',
    'respiratory_intensity', 'gastrointestinal_intensity', 'fever_intensity',
    'bleeding_intensity', 'swelling_intensity',
    'Combined_Text'
]

def prepare_model_input(engineered: dict) -> pd.DataFrame:
    X = pd.DataFrame([engineered])
    # add missing cols with defaults
    for col in REQUIRED_COLUMNS:
        if col not in X.columns:
            if col in ('Gender', 'Age_Group', 'Duration_Group'):
                X[col] = 'Unknown'
            elif col == 'Combined_Text':
                X[col] = ''
            else:
                X[col] = 0
    # numeric conversion for numeric columns
    numeric_cols = [c for c in REQUIRED_COLUMNS if c not in ('Gender', 'Age_Group', 'Duration_Group', 'Combined_Text')]
    for col in numeric_cols:
        X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
    # reorder
    X = X[REQUIRED_COLUMNS]
    return X

def predict_distribution(X: pd.DataFrame):
    # model_pipeline.classes_ are encoded labels (e.g., [0,1,2,3,4])
    proba = model_pipeline.predict_proba(X)[0]  # shape (n_classes,)
    classes_encoded = model_pipeline.classes_
    # Convert encoded to human readable via label_encoder
    try:
        decoded_labels = label_encoder.inverse_transform(classes_encoded)
    except Exception:
        # fallback: if label_encoder expects integers differently, try mapping one-by-one
        decoded_labels = []
        for enc in classes_encoded:
            try:
                decoded_labels.append(label_encoder.inverse_transform([enc])[0])
            except Exception:
                decoded_labels.append(str(enc))
    dist = {label: float(round(p * 100, 2)) for label, p in zip(decoded_labels, proba)}
    dist_sorted = dict(sorted(dist.items(), key=lambda x: x[1], reverse=True))
    return dist_sorted, proba, decoded_labels

# ---------------------------
# Maps helpers
# ---------------------------
def find_and_rank_clinics(api_key, patient_location, specialist, max_results=10):
    if not patient_location or patient_location == "Unknown":
        return {"error": "Patient location unknown, cannot search nearby clinics."}, None

    # Geocode
    geo_resp = requests.get("https://maps.googleapis.com/maps/api/geocode/json",
                            params={'address': patient_location, 'key': api_key}).json()
    if geo_resp.get('status') != 'OK':
        return {"error": f"Geocode failed: {geo_resp.get('status')}"}, None
    coords = geo_resp['results'][0]['geometry']['location']
    origin = f"{coords['lat']},{coords['lng']}"
    user_coords = {"lat": coords['lat'], "lng": coords['lng']}

    # Places text search
    places_resp = requests.get("https://maps.googleapis.com/maps/api/place/textsearch/json",
                               params={'query': f"{specialist} near {patient_location}", 'key': api_key}).json()
    if places_resp.get('status') not in ('OK', 'ZERO_RESULTS'):
        return {"error": f"Places API error: {places_resp.get('status')}"}, user_coords

    places = places_resp.get('results', [])
    clinics = []
    for p in places:
        loc = p['geometry']['location']
        clinics.append({
            "name": p.get('name'),
            "address": p.get('formatted_address'),
            "lat": loc['lat'],
            "lng": loc['lng'],
            "latlng": f"{loc['lat']},{loc['lng']}"
        })
    if not clinics:
        return [], user_coords

    # Distance matrix
    dests = "|".join([c['latlng'] for c in clinics])
    dist_resp = requests.get("https://maps.googleapis.com/maps/api/distancematrix/json",
                             params={'origins': origin, 'destinations': dests, 'mode': 'driving', 'key': api_key}).json()
    if dist_resp.get('status') != 'OK':
        return {"error": f"Distance Matrix error: {dist_resp.get('status')}"}, user_coords

    elements = dist_resp['rows'][0].get('elements', [])
    for i, el in enumerate(elements):
        if el.get('status') == 'OK':
            distance_m = el['distance']['value']
            clinics[i].update({
                "distance_text": el['distance']['text'],
                "distance_m": distance_m,
                "distance_km": round(distance_m / 1000, 2),
                "duration_text": el['duration']['text']
            })
        else:
            clinics[i].update({
                "distance_text": "N/A",
                "distance_m": float('inf'),
                "distance_km": float('inf'),
                "duration_text": "N/A"
            })

    clinics = sorted(clinics, key=lambda x: x['distance_m'])[:max_results]
    return clinics, user_coords


# ---------------------------
# Dynamic referral API: wait-time + distance + cost ranking
# ---------------------------
@app.route("/api/dynamic-referral", methods=['POST'])
def dynamic_referral():
    if wait_time_model is None:
        return jsonify({"error": "Wait time model not loaded"}), 503
    if not SNOWFLAKE_REQUIRED:
        return jsonify({"error": "Snowflake not configured"}), 503

    try:
        payload = request.get_json()
        location_text = payload.get("location_text")
        specialty = payload.get("specialty")

        if not all([location_text, specialty]):
            return jsonify({"error": "Missing location_text or specialty"}), 400

        geo_resp = requests.get(
            "https://maps.googleapis.com/maps/api/geocode/json",
            params={'address': location_text, 'key': GOOGLE_PLACES_API_KEY}
        ).json()

        if geo_resp.get('status') != 'OK':
            return jsonify({"error": f"Could not find coordinates for location: {location_text}"}), 404

        coords = geo_resp['results'][0]['geometry']['location']
        patient_lat, patient_lon = coords['lat'], coords['lng']

        conn = get_connection()
        try:
            cur = conn.cursor()
            cur.execute("SELECT * FROM specialists WHERE SPECIALTY = %s", (specialty,))
            specialists_df = df_from_cursor(cur)
        finally:
            try:
                cur.close()
            except Exception:
                pass
            conn.close()

        if specialists_df.empty:
            return jsonify({"message": f"No specialists found for {specialty}"}), 404

        # Queue lengths: mock via random for now
        np_random = np.random.randint(0, 16, size=len(specialists_df))
        specialists_df['queue_length'] = np_random

        for col in ['AVG_CONSULTATION_MINUTES', 'CAPACITY', 'CONSULTATION_FEE', 'LATITUDE', 'LONGITUDE']:
            if col not in specialists_df.columns:
                specialists_df[col] = 0
        specialists_df[['AVG_CONSULTATION_MINUTES', 'CAPACITY', 'CONSULTATION_FEE']] = (
            specialists_df[['AVG_CONSULTATION_MINUTES', 'CAPACITY', 'CONSULTATION_FEE']]
            .apply(pd.to_numeric, errors='coerce').fillna(0)
        )

        features_for_prediction = specialists_df[['queue_length', 'AVG_CONSULTATION_MINUTES', 'CAPACITY']].copy()
        features_for_prediction.columns = ['QUEUE_LENGTH', 'AVG_CONSULTATION_MINUTES', 'CAPACITY']
        specialists_df['predicted_wait_time'] = wait_time_model.predict(features_for_prediction)

        patient_origin = f"{patient_lat},{patient_lon}"
        dest_list = specialists_df.apply(lambda row: f"{row['LATITUDE']},{row['LONGITUDE']}", axis=1).tolist()

        distances_m = []
        distances_text = []
        durations_text = []
        for i in range(0, len(dest_list), 25):
            batch = "|".join(dest_list[i:i+25])
            dist_resp = requests.get(
                "https://maps.googleapis.com/maps/api/distancematrix/json",
                params={'origins': patient_origin, 'destinations': batch, 'key': GOOGLE_PLACES_API_KEY}
            ).json()

            if dist_resp.get('status') == 'OK' and dist_resp['rows'] and dist_resp['rows'][0]['elements']:
                for el in dist_resp['rows'][0]['elements']:
                    if el.get('status') == 'OK':
                        distances_m.append(el['distance']['value'])
                        distances_text.append(el['distance']['text'])
                        durations_text.append(el['duration']['text'])
                    else:
                        distances_m.append(None)
                        distances_text.append(None)
                        durations_text.append(None)
            else:
                n = len(dest_list[i:i+25])
                distances_m.extend([None] * n)
                distances_text.extend([None] * n)
                durations_text.extend([None] * n)

        if len(distances_m) < len(specialists_df):
            pad = (len(specialists_df) - len(distances_m))
            distances_m += [None] * pad
            distances_text += [None] * pad
            durations_text += [None] * pad

        specialists_df['distance_m'] = distances_m[:len(specialists_df)]
        specialists_df['distance_text'] = distances_text[:len(specialists_df)]
        specialists_df['duration_text'] = durations_text[:len(specialists_df)]

        df_nonnull = specialists_df.dropna(subset=['distance_m'])
        if df_nonnull.empty:
            min_dist = max_dist = 0
        else:
            min_dist, max_dist = df_nonnull['distance_m'].min(), df_nonnull['distance_m'].max()

        min_wait, max_wait = specialists_df['predicted_wait_time'].min(), specialists_df['predicted_wait_time'].max()
        min_cost, max_cost = specialists_df['CONSULTATION_FEE'].min(), specialists_df['CONSULTATION_FEE'].max()

        def normalize_value(value, min_val, max_val, lower_is_better=True):
            if value is None or min_val is None or max_val is None:
                return 0.0
            try:
                v = float(value); vmin = float(min_val); vmax = float(max_val)
            except Exception:
                return 0.0
            if np.isinf(v) or np.isnan(v) or np.isnan(vmin) or np.isnan(vmax):
                return 0.0
            if vmax == vmin:
                return 1.0 if lower_is_better else 0.0
            normalized = (v - vmin) / (vmax - vmin)
            normalized = max(0.0, min(1.0, normalized))
            return 1.0 - normalized if lower_is_better else normalized

        w_wait, w_dist, w_cost = 0.4, 0.4, 0.2
        specialists_df['score_wait'] = specialists_df['predicted_wait_time'].apply(
            lambda x: normalize_value(x, min_wait, max_wait, lower_is_better=True)
        )
        specialists_df['score_dist'] = specialists_df['distance_m'].apply(
            lambda x: normalize_value(x, min_dist, max_dist, lower_is_better=True) if x is not None else 0.0
        )
        specialists_df['score_cost'] = specialists_df['CONSULTATION_FEE'].apply(
            lambda x: normalize_value(x, min_cost, max_cost, lower_is_better=True)
        )

        specialists_df['final_score'] = (
            w_wait * specialists_df['score_wait'] +
            w_dist * specialists_df['score_dist'] +
            w_cost * specialists_df['score_cost']
        )

        ranked_specialists = specialists_df.sort_values(by='final_score', ascending=False)
        out_cols = [
            'SPECIALIST_ID', 'SPECIALTY', 'NAME', 'CONSULTATION_FEE',
            'AVG_CONSULTATION_MINUTES', 'CAPACITY', 'queue_length',
            'predicted_wait_time', 'distance_m', 'distance_text', 'duration_text', 'score_wait', 'score_dist', 'score_cost', 'final_score',
            'LATITUDE', 'LONGITUDE'
        ]
        present_cols = [c for c in out_cols if c in ranked_specialists.columns]
        # Add derived days column for wait time
        ranked_specialists['predicted_wait_days'] = ranked_specialists['predicted_wait_time'].apply(
            lambda m: round(float(m) / 1440.0, 2) if m is not None else None
        )
        present_cols = [c for c in out_cols + ['predicted_wait_days'] if c in ranked_specialists.columns]
        results = ranked_specialists[present_cols].to_dict('records')

        # Sanitize NaN/Infinity for JSON
        def clean_json(data):
            if isinstance(data, dict):
                return {k: clean_json(v) for k, v in data.items()}
            elif isinstance(data, list):
                return [clean_json(v) for v in data]
            elif isinstance(data, float):
                if math.isnan(data) or math.isinf(data):
                    return None
            return data

        results = clean_json(results)
        return jsonify(results)

    except Exception as e:
        app.logger.error(f"Error in /api/dynamic-referral: {e}")
        return jsonify({"error": "An internal error occurred.", "detail": str(e)}), 500

# ---------------------------
# Chatbot routes
# ---------------------------
@app.route("/api/chatbot/chat", methods=["POST"])
def chatbot_chat():
    """Handle chat with Dr. Ellie"""
    try:
        data = request.get_json()
        user_message = data.get('message', '')
        conversation_history = data.get('conversation_history', [])
        
        if not user_message:
            return jsonify({"error": "No message provided"}), 400
        
        # Get the chatbot pipeline
        qa_chain = build_chatbot_pipeline()
        
        if qa_chain is None:
            # Fallback response if pipeline fails
            fallback_responses = {
                "referral": "I can help you understand medical referrals! A referral is when your doctor recommends you see a specialist for further care.",
                "appointment": "To book an appointment, you can contact the clinic directly or use the booking buttons on your referral cards.",
                "specialist": "Specialists are doctors who focus on specific areas of medicine, like cardiology for heart issues or orthopedics for bone problems.",
                "default": "I'm here to help guide you through the MedRef system! Feel free to ask me about referrals, appointments, or how to navigate your healthcare journey."
            }
            
            # Simple keyword matching for fallback
            message_lower = user_message.lower()
            if any(word in message_lower for word in ["referral", "refer"]):
                response = fallback_responses["referral"]
            elif any(word in message_lower for word in ["appointment", "book", "schedule"]):
                response = fallback_responses["appointment"]
            elif any(word in message_lower for word in ["specialist", "doctor"]):
                response = fallback_responses["specialist"]
            else:
                response = fallback_responses["default"]
                
            return jsonify({"answer": response})
        
        # Use RAG pipeline
        try:
            # Format conversation history for the chain
            chat_history = []
            for msg in conversation_history[-10:]:  # Keep last 10 messages for context
                if msg.get('role') == 'user':
                    chat_history.append(('human', msg.get('content', '')))
                elif msg.get('role') == 'assistant':
                    chat_history.append(('ai', msg.get('content', '')))
            
            result = qa_chain({"question": user_message, "chat_history": chat_history})
            response = result["answer"]
            
        except Exception as e:
            print(f"Error with RAG pipeline: {e}")
            response = "I'm having a moment of technical difficulty, but I'm still here to help! Could you try rephrasing your question?"
        
        return jsonify({"answer": response})
        
    except Exception as e:
        print(f"Error in chatbot_chat: {e}")
        return jsonify({"error": "Internal server error"}), 500


@app.route("/api/patient/dashboard_data", methods=['GET'])
def get_patient_dashboard_data():
    """
    API endpoint to fetch all referrals for the logged-in patient from Snowflake.
    """
    if "user" not in session:
        return jsonify({"error": "Unauthorized"}), 401

    user_id = session['user'].get('id')
    if not user_id:
        return jsonify({"error": "User ID not found in session"}), 400

    conn = get_connection()
    cursor = conn.cursor(snowflake.connector.DictCursor)
    
    try:
        # CORRECTED: Explicitly select columns and use '?' for the parameter
        cursor.execute(
            """
            SELECT 
                REFERRAL_ID,
                RECOMMENDED_SPECIALIST,
                EXPLANATION,
                REFERRAL_TIMESTAMP
            FROM referrals 
            WHERE patient_user_id = %s 
            ORDER BY referral_timestamp DESC
            """, 
            (user_id,)
        )
        referrals = cursor.fetchall()

        return jsonify({
            "referrals": referrals
        })
    except Exception as e:
        # It's good practice to log the actual error on the server
        print(f"Error loading referrals for patient dashboard: {e}")
        return jsonify({"error": "Failed to load referral data."}), 500
    finally:
        cursor.close()
        conn.close()
        

@app.route("/api/specialists/<int:specialist_id>/availability", methods=['GET'])
@jwt_required()
def get_specialist_availability(specialist_id):
    """Generates fake availability for a specialist for the next 7 days."""
    conn = get_connection()
    cursor = conn.cursor()
    try:
        # Get already booked slots to exclude them
        cursor.execute("SELECT appointment_datetime FROM appointments WHERE specialist_id = %s", (specialist_id,))
        booked_slots = {row[0] for row in cursor.fetchall()}

        available_slots = []
        today = date.today()
        for i in range(7): # Generate for the next 7 days
            day = today + timedelta(days=i)
            # Generate slots from 9 AM to 5 PM, every 30 mins
            for hour in range(9, 17):
                for minute in [0, 30]:
                    slot = datetime.combine(day, time(hour, minute))
                    if slot not in booked_slots:
                        available_slots.append(slot.isoformat())
        
        return jsonify(available_slots)
    finally:
        cursor.close()
        conn.close()

@app.route("/api/appointments/book", methods=['POST'])
@jwt_required()
def book_appointment():
    """Books an appointment for the logged-in patient."""
    payload = request.get_json()
    specialist_id = payload.get('specialist_id')
    slot = payload.get('slot')
    
    current_user = get_jwt_identity()
    user_id = current_user['id']

    conn = get_connection()
    cursor = conn.cursor()
    try:
        # Simple check for double booking (a more robust check is needed in production)
        cursor.execute("SELECT 1 FROM appointments WHERE specialist_id = %s AND appointment_datetime = %s", (specialist_id, slot))
        if cursor.fetchone():
            return jsonify({"error": "This slot is no longer available."}), 409

        cursor.execute(
            "INSERT INTO appointments (patient_user_id, specialist_id, appointment_datetime) VALUES (%s, %s, %s)",
            (user_id, specialist_id, slot)
        )
        conn.commit()
        return jsonify({"message": "Appointment booked successfully!"}), 201
    finally:
        cursor.close()
        conn.close()

@app.route("/api/chatbot/ask", methods=["POST"])
def chatbot_ask():
    """Handle chatbot questions"""
    try:
        data = request.get_json()
        question = data.get("question", "").lower()
        
        if not question:
            return jsonify({"answer": "Please provide a question."})
            
        # Fallback responses when RAG pipeline is not available
        if qa_chain is None:
            return get_fallback_response(question)
            
        result = qa_chain({"question": question})
        answer = result["answer"]
        return jsonify({"answer": answer})
        
    except Exception as e:
        print(f"Error in chatbot: {e}")
        return jsonify({"answer": "I'm sorry, I encountered an error. Please try again."})

@app.route('/api/voice/transcribe', methods=['POST'])
def transcribe_voice():
    """Transcribe audio using Groq API with Whisper"""
    try:
        # Check if audio file is present
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        audio_file = request.files['audio']
        if audio_file.filename == '':
            return jsonify({'error': 'No audio file selected'}), 400
        
        # Get Groq API key
        groq_api_key = os.getenv('GROQ_API_KEY')
        if not groq_api_key:
            return jsonify({'error': 'Groq API key not configured'}), 500
        
        # Initialize Groq client
        client = Groq(api_key=groq_api_key)
        
        # Read audio file
        audio_data = audio_file.read()
        
        # Create a BytesIO object with the audio data
        audio_io = io.BytesIO(audio_data)
        
        # Set the filename for the API call
        audio_io.name = audio_file.filename
        
        # Transcribe using Groq's Whisper API
        transcription = client.audio.transcriptions.create(
            file=audio_io,
            model="whisper-large-v3",
            language="en",
            response_format="text"
        )
        
        return jsonify({
            'transcript': transcription,
            'success': True
        })
        
    except Exception as e:
        print(f"Error in voice transcription: {e}")
        # Return more specific error information
        error_msg = str(e)
        if "file must be one of the following types" in error_msg:
            return jsonify({'error': 'Unsupported audio format. Please try recording again.'}), 400
        elif "API key" in error_msg:
            return jsonify({'error': 'API configuration error'}), 500
        else:
            return jsonify({'error': 'Failed to transcribe audio'}), 500

def get_fallback_response(question):
    """Provide fallback responses when RAG pipeline is not available"""
    # Common questions and responses
    responses = {
        "how": "To use MedReferral, first go to the Referral Tool page and enter patient information. The AI will analyze the data and recommend the best specialist for your patient. Then you can view ranked clinics with wait times, distances, and costs.",
        "what": "MedReferral is an AI-powered platform that helps healthcare professionals make better referral decisions by analyzing patient data and recommending appropriate specialists. It reduces inappropriate referrals and improves patient outcomes.",
        "where": "You can find the Referral Tool in the main navigation. Click on 'Referral Tool' to start analyzing patient cases, or 'Clinic Rankings' to view specialist recommendations.",
        "when": "You can use MedReferral anytime to get specialist recommendations. The system provides real-time analysis of patient data and current wait times at different clinics.",
        "why": "MedReferral helps reduce inappropriate referrals, improve patient outcomes, and optimize healthcare resource allocation. It saves time and ensures patients get to the right specialist quickly.",
        "referral": "The referral process is simple: enter patient information in the Referral Tool, get AI recommendations with confidence scores, then view ranked clinics based on wait time, distance, and cost.",
        "specialist": "MedReferral can recommend various specialists including cardiologists, neurologists, dermatologists, orthopedists, and many others based on patient symptoms and conditions.",
        "clinic": "After getting specialist recommendations, you can view ranked clinics with detailed information about wait times (in days), distances, consultation fees, and get directions.",
        "wait": "Wait times are displayed in days and are calculated based on current queue lengths and historical data for each clinic. This helps you choose clinics with shorter wait times.",
        "cost": "Consultation fees are displayed for each clinic so you can compare costs across different providers. This helps with cost-effective referrals.",
        "criticality": "Criticality levels indicate confidence in recommendations: Green (80%+) is high confidence, Yellow (60-79%) is medium, and Red (<60%) is lower confidence requiring more evaluation.",
        "help": "I can help you understand how to use MedReferral, explain the referral process, or answer questions about the platform features. What specific area would you like help with?",
        "login": "If you're having trouble logging in, check your credentials or try resetting your password. Make sure you're using the correct email and password combination.",
        "error": "If you encounter errors, try refreshing the page or clearing your browser cache. For persistent issues, contact our support team or try using a different browser.",
        "start": "To get started with MedReferral, go to the Referral Tool page and enter your patient's information. The AI will analyze the data and provide specialist recommendations with confidence scores.",
        "analysis": "The AI analysis looks at patient symptoms, medical history, and current condition to recommend the most appropriate specialist. It considers multiple factors to ensure accurate referrals.",
        "ranking": "Clinics are ranked based on a combination of wait time, distance from patient location, and consultation cost. The system provides the best balance of these factors.",
        "confidence": "Confidence scores show how certain the AI is about its recommendations. Higher scores mean more reliable suggestions, while lower scores may need additional clinical review."
    }
    
    # Find the best matching response
    for keyword, response in responses.items():
        if keyword in question:
            return jsonify({"answer": response})
    
    # Default response for unmatched questions
    return jsonify({"answer": "I'd be happy to help! You can ask me about how to use MedReferral, the referral process, specialist recommendations, clinic rankings, wait times, costs, or any other features of the platform. What would you like to know?"})


# ---------------------------
# Helper: dynamic clinics for ranking page using LLM location
# ---------------------------
def compute_dynamic_clinics_for_map(location_text: str, specialty: str, max_results: int = 10):
    if wait_time_model is None or not SNOWFLAKE_REQUIRED:
        return None, None

    geo_resp = requests.get(
        "https://maps.googleapis.com/maps/api/geocode/json",
        params={'address': location_text, 'key': GOOGLE_PLACES_API_KEY}
    ).json()
    if geo_resp.get('status') != 'OK':
        return None, None
    coords = geo_resp['results'][0]['geometry']['location']
    user_coords = {"lat": coords['lat'], "lng": coords['lng']}
    patient_origin = f"{coords['lat']},{coords['lng']}"

    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute("SELECT * FROM specialists WHERE SPECIALTY = %s", (specialty,))
        specialists_df = df_from_cursor(cur)
    finally:
        try:
            cur.close()
        except Exception:
            pass
        conn.close()

    if specialists_df.empty:
        return [], user_coords

    np_random = np.random.randint(0, 16, size=len(specialists_df))
    specialists_df['queue_length'] = np_random
    for col in ['AVG_CONSULTATION_MINUTES', 'CAPACITY', 'CONSULTATION_FEE', 'LATITUDE', 'LONGITUDE', 'NAME', 'ADDRESS']:
        if col not in specialists_df.columns:
            specialists_df[col] = 0 if col not in ['NAME', 'ADDRESS'] else ''
    specialists_df[['AVG_CONSULTATION_MINUTES', 'CAPACITY', 'CONSULTATION_FEE']] = (
        specialists_df[['AVG_CONSULTATION_MINUTES', 'CAPACITY', 'CONSULTATION_FEE']]
        .apply(pd.to_numeric, errors='coerce').fillna(0)
    )

    features = specialists_df[['queue_length', 'AVG_CONSULTATION_MINUTES', 'CAPACITY']].copy()
    features.columns = ['QUEUE_LENGTH', 'AVG_CONSULTATION_MINUTES', 'CAPACITY']
    specialists_df['predicted_wait_time'] = wait_time_model.predict(features)

    dest_list = specialists_df.apply(lambda row: f"{row['LATITUDE']},{row['LONGITUDE']}", axis=1).tolist()
    distances_m = []
    distances_text = []
    durations_text = []
    for i in range(0, len(dest_list), 25):
        batch = "|".join(dest_list[i:i+25])
        dist_resp = requests.get(
            "https://maps.googleapis.com/maps/api/distancematrix/json",
            params={'origins': patient_origin, 'destinations': batch, 'key': GOOGLE_PLACES_API_KEY}
        ).json()
        if dist_resp.get('status') == 'OK' and dist_resp['rows'] and dist_resp['rows'][0]['elements']:
            for el in dist_resp['rows'][0]['elements']:
                if el.get('status') == 'OK':
                    distances_m.append(el['distance']['value'])
                    distances_text.append(el['distance']['text'])
                    durations_text.append(el['duration']['text'])
                else:
                    distances_m.append(None)
                    distances_text.append(None)
                    durations_text.append(None)
        else:
            n = len(dest_list[i:i+25])
            distances_m.extend([None] * n)
            distances_text.extend([None] * n)
            durations_text.extend([None] * n)

    if len(distances_m) < len(specialists_df):
        pad = (len(specialists_df) - len(distances_m))
        distances_m += [None] * pad
        distances_text += [None] * pad
        durations_text += [None] * pad

    specialists_df['distance_m'] = distances_m[:len(specialists_df)]
    specialists_df['distance_text'] = distances_text[:len(specialists_df)]
    specialists_df['duration_text'] = durations_text[:len(specialists_df)]

    df_nonnull = specialists_df.dropna(subset=['distance_m'])
    if df_nonnull.empty:
        min_dist = max_dist = 0
    else:
        min_dist, max_dist = df_nonnull['distance_m'].min(), df_nonnull['distance_m'].max()
    min_wait, max_wait = specialists_df['predicted_wait_time'].min(), specialists_df['predicted_wait_time'].max()
    min_cost, max_cost = specialists_df['CONSULTATION_FEE'].min(), specialists_df['CONSULTATION_FEE'].max()

    def normalize_value(value, min_val, max_val, lower_is_better=True):
        if value is None or min_val is None or max_val is None:
            return 0.0
        try:
            v = float(value); vmin = float(min_val); vmax = float(max_val)
        except Exception:
            return 0.0
        if np.isinf(v) or np.isnan(v) or np.isnan(vmin) or np.isnan(vmax):
            return 0.0
        if vmax == vmin:
            return 1.0 if lower_is_better else 0.0
        normalized = (v - vmin) / (vmax - vmin)
        normalized = max(0.0, min(1.0, normalized))
        return 1.0 - normalized if lower_is_better else normalized

    w_wait, w_dist, w_cost = 0.4, 0.4, 0.2
    specialists_df['score_wait'] = specialists_df['predicted_wait_time'].apply(
        lambda x: normalize_value(x, min_wait, max_wait, lower_is_better=True)
    )
    specialists_df['score_dist'] = specialists_df['distance_m'].apply(
        lambda x: normalize_value(x, min_dist, max_dist, lower_is_better=True) if x is not None else 0.0
    )
    specialists_df['score_cost'] = specialists_df['CONSULTATION_FEE'].apply(
        lambda x: normalize_value(x, min_cost, max_cost, lower_is_better=True)
    )
    specialists_df['final_score'] = (
        w_wait * specialists_df['score_wait'] +
        w_dist * specialists_df['score_dist'] +
        w_cost * specialists_df['score_cost']
    )

    ranked = specialists_df.sort_values(by='final_score', ascending=False).head(max_results)
    clinics = []
    a=14
    for _, row in ranked.iterrows():
        dm = row.get('distance_m')
        clinics.append({
            "name": row.get('NAME') or f"{specialty} Specialist",
            "address": row.get('ADDRESS') or "",
            "lat": float(row.get('LATITUDE') or 0),
            "lng": float(row.get('LONGITUDE') or 0),
            "latlng": f"{row.get('LATITUDE')},{row.get('LONGITUDE')}",
            "distance_text": row.get('distance_text') or "",
            "distance_m": dm if dm is not None else float('inf'),
            "distance_km": round((dm or 0) / 1000, 2) if dm is not None else float('inf'),
            "duration_text": row.get('duration_text') or "",
            "cost": float(row.get('CONSULTATION_FEE') or 0),
            "predicted_wait_days": float(a)
        })
        a+=random.randint(2,5)
    return clinics, user_coords

# ---------------------------
# Routes: landing, auth, referral, specialist, ranking, api
# ---------------------------
@app.route("/")
def landing():
    return render_template("index.html")

# ---- Signup / Login (Snowflake) ----
@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        role = "clinician"

        # Basic validations (can be extended)
        errors = {}
        if not re.match(r"^[A-Za-z0-9_]{3,30}$", username):
            errors['username'] = "Username must be 3-30 chars (letters, numbers, underscore)."
        if not re.match(r"^[\w\.-]+@[\w\.-]+\.\w+$", email):
            errors['email'] = "Invalid email."
        if len(password) < 8:
            errors['password'] = "Password must be at least 8 characters."

        if errors:
            return render_template("signup.html", errors=errors, username=username, email=email)

        hashed = generate_password_hash(password)
        if SNOWFLAKE_REQUIRED:
            try:
                conn = get_connection()
                cur = conn.cursor()
                # Try inserting role if the column exists; otherwise insert without role
                try:
                    cur.execute("INSERT INTO users (username, email, password_hash, role) VALUES (%s, %s, %s, %s)",
                                (username, email, hashed, role))
                except Exception:
                    cur.execute("INSERT INTO users (username, email, password_hash) VALUES (%s, %s, %s)",
                                (username, email, hashed))
                    # Persist role mapping locally for Snowflake users when role column not present
                    roles_map = _load_local_roles(); roles_map[email] = role; _save_local_roles(roles_map)
                conn.commit()
                cur.close()
                conn.close()
                flash("Account created — please log in.", "success")
                return redirect(url_for("login"))
            except Exception as e:
                # handle unique constraint / other db errors
                msg = str(e)
                if "unique" in msg.lower():
                    flash("Username or email already exists.", "danger")
                else:
                    flash(f"Signup failed: {msg}", "danger")
                return render_template("signup.html", username=username, email=email)
        else:
            # Local fallback user creation
            users = _load_local_users()
            # Uniqueness checks
            if any(u.get("username") == username for u in users) or any(u.get("email") == email for u in users):
                flash("Username or email already exists.", "danger")
                return render_template("signup.html", username=username, email=email)
            local_user = {
                "id": _next_local_user_id(users),
                "username": username,
                "email": email,
                "password_hash": hashed,
                "role": role
            }
            users.append(local_user)
            try:
                _save_local_users(users)
                flash("Account created — please log in.", "success")
                return redirect(url_for("login"))
            except Exception as e:
                flash(f"Signup failed: {e}", "danger")
                return render_template("signup.html", username=username, email=email)
    return render_template("signup.html")

@app.route("/patient/signup", methods=["GET", "POST"])
def patient_signup():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        role = "patient"

        errors = {}
        if not re.match(r"^[A-Za-z0-9_]{3,30}$", username):
            errors['username'] = "Username must be 3-30 chars (letters, numbers, underscore)."
        if not re.match(r"^[\w\.-]+@[\w\.-]+\.\w+$", email):
            errors['email'] = "Invalid email."
        if len(password) < 8:
            errors['password'] = "Password must be at least 8 characters."
        if errors:
            return render_template("patient_signup.html", errors=errors, username=username, email=email)

        hashed = generate_password_hash(password)
        if SNOWFLAKE_REQUIRED:
            try:
                conn = get_connection()
                cur = conn.cursor()
                try:
                    cur.execute("INSERT INTO users (username, email, password_hash, role) VALUES (%s, %s, %s, %s)",
                                (username, email, hashed, role))
                except Exception:
                    cur.execute("INSERT INTO users (username, email, password_hash) VALUES (%s, %s, %s)",
                                (username, email, hashed))
                    roles_map = _load_local_roles(); roles_map[email] = role; _save_local_roles(roles_map)
                conn.commit()
                cur.close(); conn.close()
                flash("Account created — please log in.", "success")
                return redirect(url_for("patient_login"))
            except Exception as e:
                msg = str(e)
                if "unique" in msg.lower():
                    flash("Username or email already exists.", "danger")
                else:
                    flash(f"Signup failed: {msg}", "danger")
                return render_template("patient_signup.html", username=username, email=email)
        else:
            users = _load_local_users()
            if any(u.get("username") == username for u in users) or any(u.get("email") == email for u in users):
                flash("Username or email already exists.", "danger")
                return render_template("patient_signup.html", username=username, email=email)
            local_user = {
                "id": _next_local_user_id(users),
                "username": username,
                "email": email,
                "password_hash": hashed,
                "role": role
            }
            users.append(local_user)
            try:
                _save_local_users(users)
                flash("Account created — please log in.", "success")
                return redirect(url_for("patient_login"))
            except Exception as e:
                flash(f"Signup failed: {e}", "danger")
                return render_template("patient_signup.html", username=username, email=email)
    return render_template("patient_signup.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        # Support both form-encoded and JSON (AJAX) submissions
        if request.is_json:
            payload = request.get_json(silent=True) or {}
            email = (payload.get("email") or "").strip().lower()
            password = payload.get("password") or ""
            wants_json = True
        else:
            email = request.form.get("email", "").strip().lower()
            password = request.form.get("password", "")
            wants_json = request.headers.get("X-Requested-With") == "XMLHttpRequest"

        if SNOWFLAKE_REQUIRED:
            try:
                conn = get_connection()
                cur = conn.cursor()
                cur.execute("SELECT id, username, email, password_hash FROM users WHERE email=%s", (email,))
                row = cur.fetchone()
                cur.close()
                conn.close()
            except Exception as e:
                if wants_json:
                    return jsonify({
                        "success": False,
                        "message": f"Login DB error: {e}"
                    }), 500
                flash(f"Login DB error: {e}", "danger")
                return render_template("login.html")

            if row and check_password_hash(row[3], password):
                # Fresh session to prevent fixation
                session.clear()
                user_identity = {
                    "id": row[0],
                    "username": row[1],
                    "email": row[2]
                }
                # Determine role from DB if available; else from local role mapping; default clinician
                try:
                    # If SELECT included role column in index 4
                    if len(row) > 4 and row[4]:
                        user_identity["role"] = row[4]
                except Exception:
                    pass
                if "role" not in user_identity:
                    roles_map = _load_local_roles()
                    role_guess = roles_map.get(user_identity["email"], "clinician")
                    user_identity["role"] = role_guess

                session['user'] = user_identity
                session['logged_in'] = True
                session['user_role'] = user_identity.get('role', 'clinician')
                session.permanent = True

                # Optional: JWT for API-only usage
                access_token = create_access_token(identity=user_identity)
                session['access_token'] = access_token

                if wants_json:
                    return jsonify({
                        "success": True,
                        "access_token": access_token,
                        "redirect_url": url_for("referral")
                    })
                flash("Login successful", "success")
                return redirect(url_for("referral"))
            else:
                if wants_json:
                    return jsonify({
                        "success": False,
                        "message": "Invalid credentials"
                    }), 401
                flash("Invalid credentials", "danger")
                return render_template("login.html")
        else:
            # Local fallback authentication
            users = _load_local_users()
            user = next((u for u in users if u.get("email") == email), None)
            if user and check_password_hash(user.get("password_hash", ""), password):
                session.clear()
                user_identity = {
                    "id": user.get("id"),
                    "username": user.get("username"),
                    "email": user.get("email"),
                    "role": user.get("role", "clinician")
                }
                session['user'] = user_identity
                session['logged_in'] = True
                session['user_role'] = user_identity.get('role', 'clinician')
                session.permanent = True
                access_token = create_access_token(identity=user_identity)
                session['access_token'] = access_token
                if wants_json:
                    return jsonify({
                        "success": True,
                        "access_token": access_token,
                        "redirect_url": url_for("referral")
                    })
                flash("Login successful", "success")
                return redirect(url_for("referral"))
            else:
                if wants_json:
                    return jsonify({
                        "success": False,
                        "message": "Invalid credentials"
                    }), 401
                flash("Invalid credentials", "danger")
                return render_template("login.html")

    return render_template("login.html")

# ---- Patient Login & Dashboard ----
@app.route("/patient/login", methods=["GET", "POST"])
def patient_login():
    if request.method == "POST":
        # Support both form and JSON
        if request.is_json:
            payload = request.get_json(silent=True) or {}
            email = (payload.get("email") or "").strip().lower()
            password = payload.get("password") or ""
            wants_json = True
        else:
            email = request.form.get("email", "").strip().lower()
            password = request.form.get("password", "")
            wants_json = request.headers.get("X-Requested-With") == "XMLHttpRequest"

        # Try Snowflake first, else local fallback
        user_row = None
        if SNOWFLAKE_REQUIRED:
            try:
                conn = get_connection()
                cur = conn.cursor()
                cur.execute("SELECT id, username, email, password_hash FROM users WHERE email=%s", (email,))
                user_row = cur.fetchone()
                cur.close(); conn.close()
            except Exception as e:
                if wants_json:
                    return jsonify({"success": False, "message": f"Login DB error: {e}"}), 500
                flash(f"Login DB error: {e}", "danger")
                return render_template("patient_login.html")
            if user_row and check_password_hash(user_row[3], password):
                session.clear()
                user_identity = {"id": user_row[0], "username": user_row[1], "email": user_row[2]}
                # Prefer stored role if available; fallback to patient for this path
                try:
                    if len(user_row) > 4 and user_row[4]:
                        user_identity["role"] = user_row[4]
                except Exception:
                    pass
                if "role" not in user_identity:
                    roles_map = _load_local_roles();
                    user_identity["role"] = roles_map.get(user_identity["email"], "patient")
                session['user'] = user_identity
                session['logged_in'] = True
                session['user_role'] = user_identity.get('role', 'patient')
                session.permanent = True
                access_token = create_access_token(identity=user_identity)
                session['access_token'] = access_token
                if wants_json:
                    return jsonify({"success": True, "access_token": access_token, "redirect_url": url_for("patient_dashboard")})
                flash("Login successful", "success")
                return redirect(url_for("patient_dashboard"))
        # Local fallback
        users = _load_local_users()
        user = next((u for u in users if u.get("email") == email), None)
        if user and check_password_hash(user.get("password_hash", ""), password):
            session.clear()
            user_identity = {"id": user.get("id"), "username": user.get("username"), "email": user.get("email"), "role": user.get("role", "patient")}
            session['user'] = user_identity
            session['logged_in'] = True
            session['user_role'] = user_identity.get('role', 'patient')
            session.permanent = True
            access_token = create_access_token(identity=user_identity)
            session['access_token'] = access_token
            if wants_json:
                return jsonify({"success": True, "access_token": access_token, "redirect_url": url_for("patient_dashboard")})
            flash("Login successful", "success")
            return redirect(url_for("patient_dashboard"))

        if wants_json:
            return jsonify({"success": False, "message": "Invalid credentials"}), 401
        flash("Invalid credentials", "danger")
        return render_template("patient_login.html")
    return render_template("patient_login.html")

@app.route("/patient/dashboard")
def patient_dashboard():
    """
    Fetches and displays all referrals for the logged-in patient.
    """
    if "user" not in session:
        flash("Please log in to view your dashboard.", "warning")
        return redirect(url_for("login"))

    # FIX 1: Define user_id and patient_email from the session
    user_id = session['user'].get('id')
    patient_email = session['user'].get('email') # This was the missing line

    if not user_id:
        flash("Could not identify user.", "danger")
        return redirect(url_for("login"))
    
    referrals = []
    try:
        conn = get_connection()
        cur = conn.cursor()
        
        # FIX 2: Use '?' for the parameter placeholder to prevent errors
        cur.execute(
            """
            SELECT 
                REFERRAL_ID,
                RECOMMENDED_SPECIALIST,
                EXPLANATION,
                REFERRAL_TIMESTAMP,
                TOP_CLINIC_NAME 
            FROM referrals 
            WHERE patient_user_id = %s 
            ORDER BY referral_timestamp DESC
            """,
            (user_id,)
        )
        rows = cur.fetchall()
        print(f"[DB INFO] Patient dashboard fetched {len(rows)} referrals for {patient_email}")

        # FIX 3: Simplify the loop to match your new table schema
        # The old code was trying to parse JSON and access columns that don't exist
        for row in rows:
            referrals.append({
                "referral_id": row[0],
                "recommended_specialist": row[1],
                "explanation": row[2],
                "created_at": str(row[3]) if row[3] is not None else None,
                "top_clinic_name": row[4] # <-- Add the new data from the database
            })
        cur.close(); conn.close()

    except Exception as e:
        print(f"Error loading referrals for patient dashboard: {e}")
        flash("Unable to load your referrals right now.", "danger")

    return render_template("patient_dashboard.html", referrals=referrals, patient_email=patient_email)

# ---- Google OAuth ----
@app.route("/google/login")
def google_login():
    redirect_uri = url_for('google_auth_callback', _external=True)
    return google.authorize_redirect(redirect_uri)

@app.context_processor
def inject_user():
    """Make user session data available to all templates"""
    return {
        'current_user': session.get('user'),
        'is_logged_in': session.get('logged_in', False),
        'now':datetime.now
    }

@app.template_filter('datetime_format')
def datetime_format(dt, format='%b %d, %Y %I:%M %p'):
    """Format datetime objects"""
    return dt.strftime(format)

# Also add a helper route to check session status
@app.route("/api/session-status")
def session_status():
    """API endpoint to check current session status"""
    return jsonify({
        'logged_in': session.get('logged_in', False),
        'user': session.get('user')
    })



@app.route("/auth/callback")
def google_auth_callback():
    try:
        token = google.authorize_access_token()
        if not token:
            flash("Google login failed (no token).", "danger")
            return redirect(url_for("login"))
            
        user_info = google.get('userinfo').json()
        email = user_info.get('email')
        oauth_id = user_info.get('id')
        username = user_info.get('name', email.split('@')[0] if email else 'user')
        
        if not email:
            flash("Google did not return email", "danger")
            return redirect(url_for("login"))

        # Handle Snowflake user creation/update (existing code)
        if SNOWFLAKE_REQUIRED:
            # ... existing Snowflake code ...
            pass

        # Create session + token
        # Fresh session to prevent fixation
        session.clear()
        user_identity = {
            "username": username,
            "email": email,
            "oauth_provider": "google"
        }

        session['user'] = user_identity
        session['logged_in'] = True
        session.permanent = True

        access_token = create_access_token(identity=user_identity)
        session['access_token'] = access_token
        
        flash("Logged in with Google", "success")
        return redirect(url_for("referral"))

    except Exception as e:
        flash(f"Google OAuth error: {e}", "danger")
        return redirect(url_for("login"))

@app.route("/logout")
def logout():
    # Clear entire session and set new cookie
    session.clear()
    flash("Logged out", "info")
    return redirect(url_for("landing"))

@app.route("/profile")
def profile():
    if not session.get('logged_in') or 'user' not in session:
        return redirect(url_for("login"))
    return render_template("profile_standalone.html", user=session.get('user'))

@app.after_request
def add_security_headers(response):
    response.headers['Cache-Control'] = 'no-store'
    response.headers['Pragma'] = 'no-cache'
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
    return response




# ---- Referral flow (LLM -> Model -> Maps) ----
@app.route("/referral", methods=["GET", "POST"])
def referral():
    # require login
    if "user" not in session:
        return redirect(url_for("login"))
    # clinicians only
    if session.get('user_role') not in (None, 'clinician'):
        flash("Access restricted to clinicians.", "danger")
        return redirect(url_for("landing"))

    if request.method == "POST":
        patient_note = request.form.get("patient_note", "").strip()
        patient_email = request.form.get("patient_email", "").strip().lower()
        if not patient_note:
            flash("Please enter patient note.", "danger")
            return redirect(url_for("referral"))
        if not patient_email:
            flash("Please enter patient email.", "danger")
            return redirect(url_for("referral"))

        # 1) Extract structured JSON from LLM
        try:
            relevance_result = relevance_chain.invoke({"note": patient_note})
            # LangChain can return dict or AIMessage, handle both
            relevance_text = (relevance_result.get("text", "") if isinstance(relevance_result, dict) else str(relevance_result.content)).strip()

            if "irrelevant" in relevance_text.lower():
                flash("The provided note was determined to be medically irrelevant.", "warning")
                return render_template("referral_result.html",
                                       is_irrelevant=True,
                                       explanation="The system could not process the request because the patient note did not contain medically relevant information for a specialist referral.")
        except Exception as e:
            flash(f"Relevance check failed: {e}", "danger")
            return redirect(url_for("referral"))

        # 1) Extract structured JSON from LLM
        try:
            raw = extract_chain.invoke({"note": patient_note})
            structured = parse_llm_output(raw.get('text') if isinstance(raw, dict) else raw.content)
        except Exception as e:
            flash(f"LLM extraction failed: {e}", "danger")
            return redirect(url_for("referral"))

        # 2) Engineer features
        engineered = engineer_features(structured)

        # 3) Prepare input DataFrame
        X = prepare_model_input(engineered)

        # 4) Predict distribution
        try:
            dist_sorted, proba_array, decoded_labels = predict_distribution(X)
        except Exception as e:
            flash(f"Model prediction failed: {e}", "danger")
            return redirect(url_for("referral"))

        # 5) Top prediction from ML model
        ml_top_label = next(iter(dist_sorted.keys()))
        top_pct = dist_sorted[ml_top_label]
        
        # Keep track of original prediction and override status
        is_overridden = False
        final_specialist = ml_top_label
        explanation = ""

        # =================================================================
        # NEW STEP 5.5: LLM Validation and Override
        # =================================================================
        try:
            specialist_list_str = ", ".join(SPECIALIST_LIST)
            validation_result_raw = validation_chain.invoke({
                "note": patient_note,
                "ml_specialist": ml_top_label,
                "specialist_list": specialist_list_str
            })
            
            validation_result = parse_validation_output(validation_result_raw.get('text') if isinstance(validation_result_raw, dict) else validation_result_raw.content)
            
            validated_specialist = validation_result.get("validated_specialist")
            
            # Check if the validated specialist is valid and different from the ML model's prediction
            if validated_specialist and validated_specialist in SPECIALIST_LIST and validated_specialist != ml_top_label:
                is_overridden = True
                final_specialist = validated_specialist
                explanation = validation_result.get("validation_reason", f"Referral corrected to {final_specialist} based on a detailed review of the patient's case.")
                flash(f"The initial recommendation ({ml_top_label}) was reviewed and updated to {final_specialist} for better accuracy.", "info")
            
        except Exception as e:
            print(f"[WARN] LLM Validation step failed: {e}. Proceeding with ML model prediction.")
            # Fallback if validation fails: just use the ML prediction
            pass

        # 6) LLM explainability (only if not overridden)
        if not is_overridden:
            try:
                # Select top numeric features for explanation
                numeric_cols = [c for c in X.columns if X[c].dtype != 'object']
                top_features_pairs = sorted([(c, float(X.iloc[0][c])) for c in numeric_cols],
                                            key=lambda x: abs(x[1]), reverse=True)[:6]
                top_features_text = "\n".join([f"{k}: {v:.2f}" for k, v in top_features_pairs])
                
                explain_result = explain_chain.invoke({
                    "note": patient_note,
                    "specialist": final_specialist,
                    "top_features": top_features_text
                })
                explanation = explain_result.get('text') if isinstance(explain_result, dict) else explain_result.content
            except Exception:
                explanation = f"Recommended {final_specialist} based on the extracted clinical features."

        # 7) Save session values (using the FINAL specialist)
        session['patient_note'] = patient_note
        session['patient_email'] = patient_email
        session['location'] = structured.get('Location', 'Unknown')
        session['top_specialist'] = final_specialist # Use the final validated/overridden specialist
        session['distribution'] = dist_sorted # The original distribution from the model
        session['explanation'] = explanation
        session['extracted_data'] = structured
        # 8) Persist referral (Snowflake or local)
        try:
            if SNOWFLAKE_REQUIRED:
                conn = get_connection()
                cur = conn.cursor()

                # Resolve doctor_user_id from session email
                doctor_user_id = None
                try:
                    cur.execute("SELECT ID FROM USERS WHERE EMAIL=%s", (session['user']['email'],))
                    r = cur.fetchone()
                    if r:
                        doctor_user_id = r[0]
                except Exception as e_lookup:
                    _log_db_error(e_lookup, "lookup doctor id by email")

                if doctor_user_id is None:
                    # You can decide whether to hard-fail or proceed with NULL (schema requires NOT NULL)
                    raise RuntimeError("Doctor user not found in USERS table for session email.")

                # Resolve patient_user_id from patient email (if patient account exists)
                patient_user_id = None
                try:
                    cur.execute("SELECT ID FROM USERS WHERE EMAIL=%s", (patient_email,))
                    r = cur.fetchone()
                    if r:
                        patient_user_id = r[0]
                except Exception as e_lookup:
                    _log_db_error(e_lookup, "lookup patient id by email")

                # Compact JSON payload to store extra details
                extra_payload = {
                    "patient_email": patient_email,
                    "patient_note": patient_note,
                    "location": session['location'],
                    "distribution": dist_sorted,
                    "extracted": structured
                }

                print("[DEBUG] session['user'] contents:", session['user'])
                clinics, _ = find_and_rank_clinics(GOOGLE_PLACES_API_KEY, session['location'], final_specialist)

                top_clinic_name = None # Default to NULL if no clinics found
                if clinics:
                    top_clinic_name = clinics[0].get('name')
                # Insert matching existing Snowflake schema
                try:
                    cur.execute(
                        """
                        INSERT INTO REFERRALS (
                            DOCTOR_USER_ID, PATIENT_USER_ID, RECOMMENDED_SPECIALIST,
                            EXPLANATION,TOP_CLINIC_NAME
                        ) VALUES (%s, %s, %s, %s, %s)
                        """,
                        (
                            doctor_user_id, patient_user_id, final_specialist,
                            explanation,top_clinic_name
                        )
                    )
                    conn.commit()
                    print(f"[DB INFO] Inserted referral row (doctor_id={doctor_user_id}, patient_user_id={patient_user_id})")
                except Exception as e_ins:
                    import traceback
                    print("[DB DEBUG] Exception type:", type(e_ins))
                    print("[DB DEBUG] Exception args:", e_ins.args)
                    print("[DB DEBUG] Traceback:")
                    traceback.print_exc()
                    _log_db_error(e_ins, "insert referrals row")
                    raise
                finally:
                    try:
                        cur.close(); conn.close()
                    except Exception:
                        pass

            else:
                # Local JSON fallback (mirror Snowflake schema field names)
                local_referrals_file = os.path.join(os.path.dirname(__file__), "local_referrals.json")
                referrals = []
                if os.path.exists(local_referrals_file):
                    with open(local_referrals_file, 'r', encoding='utf-8') as f:
                        referrals = json.load(f)

                # Derive a local autoincrement-like REFERRAL_ID
                next_id = (max([int(r.get('REFERRAL_ID', 0)) for r in referrals]) + 1) if referrals else 1

                # Try to look up doctor & patient IDs locally if you maintain a local users cache;
                # otherwise, set doctor_user_id to None (or derive from session if you store it).
                # For consistency with Snowflake, we’ll mirror the same fields:
                doctor_user_id = None  # you can store this at login in session if desired
                patient_user_id = None

                new_ref = {
                    "REFERRAL_ID": next_id,
                    "DOCTOR_USER_ID": doctor_user_id,
                    "PATIENT_USER_ID": patient_user_id,
                    "RECOMMENDED_SPECIALIST": final_specialist,
                    "EXPLANATION": explanation,
                    "EXTRACTED_DATA_JSON": {
                        "patient_email": patient_email,
                        "patient_note": patient_note,
                        "location": session['location'],
                        "distribution": dist_sorted,
                        "extracted": structured
                    },
                    "REFERRAL_TIMESTAMP": datetime.now().isoformat()
                }

                referrals.append(new_ref)
                with open(local_referrals_file, 'w', encoding='utf-8') as f:
                    json.dump(referrals, f, ensure_ascii=False, indent=2)

                print(f"[LOCAL INFO] Saved referral locally for patient_email={patient_email} total_local_referrals={len(referrals)}")

        except Exception as e:
            print(f"Warning: failed to persist referral: {e}")
            flash("Referral created but saving to storage failed.", "warning")

        # 9) Search nearby clinics (if location known)
        
        # render results
        return render_template("referral_result.html",
                               recommended=final_specialist,
                               recommended_pct=top_pct,
                               distribution=dist_sorted,
                               explanation=explanation,
                               clinics=clinics,
                               patient_location=session['location'],
                               current_time=datetime.now(),
                               extracted_data=structured,
                               patient_email=patient_email,
                               is_overridden=is_overridden, # Pass override status to template
                               original_recommendation=ml_top_label if is_overridden else None)

    # GET
    return render_template("referral.html")
@app.route("/api/patient/followup", methods=["POST"])
def patient_followup():
    """Handle patient follow-up data submission to Snowflake"""
    try:
        if "user" not in session:
            return jsonify({"error": "Not authenticated"}), 401
            
        data = request.get_json()
        referral_id = data.get("referral_id")
        specialist = data.get("specialist")
        responses = data.get("followup_responses")
        timestamp = data.get("completion_timestamp")
        user_id = session['user']['id']
        
        if not all([referral_id, responses, user_id]):
            return jsonify({"error": "Missing required data"}), 400
        
        # Connect to Snowflake
        conn = snowflake.connector.connect(
            user=os.getenv('SNOWFLAKE_USER'),
            password=os.getenv('SNOWFLAKE_PASSWORD'),
            account=os.getenv('SNOWFLAKE_ACCOUNT'),
            warehouse=os.getenv('SNOWFLAKE_WAREHOUSE'),
            database=os.getenv('SNOWFLAKE_DATABASE'),
            schema=os.getenv('SNOWFLAKE_SCHEMA')
        )
        
        cur = conn.cursor()
        
        # Insert follow-up data
        cur.execute("""
            INSERT INTO patient_followups (
                patient_user_id,
                referral_id,
                specialist_type,
                overall_feeling,
                new_medications,
                concerns,
                satisfaction_rating,
                additional_notes,
                followup_timestamp,
                status
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            user_id,
            referral_id,
            specialist,
            responses.get('overall_feeling', ''),
            responses.get('new_medications', ''),
            responses.get('concerns', ''),
            responses.get('satisfaction', ''),
            responses.get('additional_notes', ''),
            timestamp,
            'completed'
        ))
        
        # Update referral status
        cur.execute("""
            UPDATE referrals 
            SET followup_completed = TRUE, 
                needs_followup = FALSE,
                followup_completion_date = %s
            WHERE referral_id = %s AND patient_user_id = %s
        """, (timestamp, referral_id, user_id))
        
        conn.commit()
        cur.close()
        conn.close()
        
        return jsonify({"success": True, "message": "Follow-up submitted successfully"})
        
    except Exception as e:
        print(f"Error submitting followup: {e}")
        return jsonify({"error": "Failed to submit follow-up"}), 500
@app.route("/specialist/<name>")
def specialist_page(name):
    patient_note = session.get('patient_note', '')
    distribution = session.get('distribution', {})
    explanation = session.get('explanation')
    # If no explanation in session, generate one quickly
    if not explanation and patient_note:
        try:
            explanation = explain_chain.run({"note": patient_note, "specialist": name, "top_features": ""})
        except Exception:
            explanation = f"Referral: {name}."
    return render_template("specialist.html", specialist=name, explanation=explanation, distribution=distribution)

@app.route("/ranking")
def ranking():
    if "user" not in session:
        return redirect(url_for("login"))
    if session.get('user_role') not in (None, 'clinician'):
        flash("Access restricted to clinicians.", "danger")
        return redirect(url_for("landing"))
    specialist = session.get('top_specialist')
    location = session.get('location', 'Unknown')
    if not specialist:
        flash("No referral available. Run a referral first.", "warning")
        return redirect(url_for("referral"))

    # Prefer dynamic ranking if model + snowflake are available; fallback to simple distance-based
    clinics = []
    user_coords = None
    if wait_time_model is not None and SNOWFLAKE_REQUIRED and location and location != 'Unknown':
        try:
            clinics, user_coords = compute_dynamic_clinics_for_map(location, specialist, max_results=10)
        except Exception as _e:
            print(f"Warning: dynamic map ranking failed: {_e}")
    if not clinics:
        clinics, user_coords = find_and_rank_clinics(GOOGLE_PLACES_API_KEY, location, specialist)

    user_lat = user_coords["lat"] if user_coords else None
    user_lng = user_coords["lng"] if user_coords else None
    return render_template("ranking.html", clinics=clinics, specialist=specialist, location=location, user_lat=user_lat, user_lng=user_lng)

# Simple JSON API for frontend
@app.route("/api/referral", methods=["POST"])
def api_referral():
    payload = request.get_json(force=True)
    note = payload.get("patient_note", "")
    if not note:
        return jsonify({"error": "patient_note required"}), 400
    try:
        raw = extract_chain.run({"note": note})
        structured = parse_llm_output(raw)
        engineered = engineer_features(structured)
        X = prepare_model_input(engineered)
        dist_sorted, _, _ = predict_distribution(X)
        top = next(iter(dist_sorted.keys()))
        return jsonify({"recommended": top, "distribution": dist_sorted, "location": structured.get("Location", "Unknown")})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ---------------------------
# Run
# ---------------------------
if __name__ == "__main__":
    # Initialize chatbot (optional - will use fallback responses if not available)
    try:
        qa_chain = build_chatbot_pipeline()
        if qa_chain:
            print("Dr. Ellie chatbot initialized successfully! 😊")
        else:
            print("Dr. Ellie chatbot using fallback responses (RAG pipeline not available)")
    except Exception as e:
        print(f"Dr. Ellie chatbot using fallback responses due to error: {e}")
        qa_chain = None
    
    app.run(debug=True, host="0.0.0.0")