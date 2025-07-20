import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import random
import json
import asyncio
import os
from dotenv import load_dotenv # Needed if running locally with .env, harmless in Canvas

import aiohttp

# Load environment variables (only relevant for local development with a .env file)
# In the Canvas environment, API_KEY will be automatically provided even if this is not loaded from .env
load_dotenv() 

# API_KEY is intentionally set to "" here.
# When run in the Canvas environment, a valid API key for Gemini will be injected automatically.
# If running locally and you wish to use your own Gemini API key, uncomment the line below
# and ensure you have GEMINI_API_KEY="YOUR_KEY" in your .env file.
API_KEY = "AIzaSyDI-rnRDZa3jMUrdmoUxS41BhSSLJL084o" # os.getenv("GEMINI_API_KEY") 

GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

def init_granite_model():
    st.session_state['model_initialized'] = True
    return "IBM Granite model connection initialized (simulated)."

async def call_gemini_api(prompt, response_schema=None):
    headers = {'Content-Type': 'application/json'}
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "responseMimeType": "application/json" if response_schema else "text/plain",
            "responseSchema": response_schema
        }
    }
    params = {'key': API_KEY}

    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(GEMINI_API_URL, headers=headers, params=params, data=json.dumps(payload)) as response:
                response.raise_for_status()
                result = await response.json()
                
                if result.get("candidates") and result["candidates"][0].get("content") and result["candidates"][0]["content"].get("parts"):
                    text_content = result["candidates"][0]["content"]["parts"][0].get("text")
                    if response_schema:
                        return json.loads(text_content) if text_content else {}
                    return text_content
                else:
                    st.error(f"Error: Unexpected response structure from Gemini API: {result}")
                    return "Sorry, I couldn't generate a response."
        except aiohttp.ClientError as e:
            st.error(f"API call failed: {e}")
            return "Sorry, there was an error connecting to the AI assistant."
        except json.JSONDecodeError:
            st.error("Failed to decode JSON response from API.")
            return "Sorry, I received an unreadable response from the AI."
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
            return "An unexpected error occurred with the AI assistant."


async def answer_patient_query(query, chat_history):
    prompt = f"""
    As a healthcare AI assistant, provide a helpful, accurate, and evidence-based response to the following patient question:

    PATIENT QUESTION: {query}

    Provide a clear, empathetic response that:
    - Directly addresses the question
    - Includes relevant medical facts
    - Acknowledges limitations (when appropriate)
    - Suggests when to seek professional medical advice
    - Uses accessible, non-technical language

    RESPONSE:
    """
    return await call_gemini_api(prompt)

async def predict_disease(symptoms, patient_profile):
    prompt = f"""
    As a medical AI assistant, predict potential health conditions based on the following patient data:

    Current Symptoms: {symptoms}
    Age: {patient_profile.get('age', 'N/A')}
    Gender: {patient_profile.get('gender', 'N/A')}
    Medical History: {patient_profile.get('medical_history', 'None')}
    Recent Health Metrics:
    - Average Heart Rate: {patient_profile.get('avg_heart_rate', 'N/A')} bpm
    - Average Blood Pressure: {patient_profile.get('avg_bp_systolic', 'N/A')}/{patient_profile.get('avg_bp_diastolic', 'N/A')} mmHg
    - Average Blood Glucose: {patient_profile.get('avg_glucose', 'N/A')} mg/dL
    - Recently Reported Symptoms: {patient_profile.get('recent_symptoms', 'None')}

    Format your response as a JSON object with a list of potential conditions. Each condition should have a 'name', 'likelihood' (High/Medium/Low), 'explanation', and 'next_steps'. Provide the top 3 most likely conditions based on the data provided.

    Example JSON structure:
    {{
        "potential_conditions": [
            {{
                "name": "Condition A",
                "likelihood": "High",
                "explanation": "Brief explanation for A.",
                "next_steps": "Recommended next steps for A."
            }},
            {{
                "name": "Condition B",
                "likelihood": "Medium",
                "explanation": "Brief explanation for B.",
                "next_steps": "Recommended next steps for B."
            }}
        ]
    }}
    """
    schema = {
        "type": "OBJECT",
        "properties": {
            "potential_conditions": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "name": {"type": "STRING"},
                        "likelihood": {"type": "STRING", "enum": ["High", "Medium", "Low"]},
                        "explanation": {"type": "STRING"},
                        "next_steps": {"type": "STRING"}
                    },
                    "required": ["name", "likelihood", "explanation", "next_steps"]
                }
            }
        },
        "required": ["potential_conditions"]
    }
    return await call_gemini_api(prompt, response_schema=schema)


async def generate_treatment_plan(condition, patient_profile):
    prompt = f"""
    As a medical AI assistant, generate a personalized treatment plan for the following scenario:

    Patient Profile:
    - Condition: {condition}
    - Age: {patient_profile.get('age', 'N/A')}
    - Gender: {patient_profile.get('gender', 'N/A')}
    - Medical History: {patient_profile.get('medical_history', 'None')}

    Create a comprehensive, evidence-based treatment plan that includes:
    1. Recommended medications (include dosage guidelines if appropriate)
    2. Lifestyle modifications
    3. Follow-up testing and monitoring
    4. Dietary recommendations
    5. Physical activity guidelines
    6. Mental health considerations

    Format this as a clear, structured JSON object with keys for each section.

    Example JSON structure:
    {{
        "medications": ["Medication 1 (Dosage)", "Medication 2"],
        "lifestyle_modifications": ["Modification 1", "Modification 2"],
        "follow_up": ["Follow-up 1", "Follow-up 2"],
        "dietary_recommendations": ["Dietary Rec 1", "Dietary Rec 2"],
        "physical_activity": ["Activity 1", "Activity 2"],
        "mental_health_considerations": ["Consideration 1"]
    }}
    """
    schema = {
        "type": "OBJECT",
        "properties": {
            "medications": {"type": "ARRAY", "items": {"type": "STRING"}},
            "lifestyle_modifications": {"type": "ARRAY", "items": {"type": "STRING"}},
            "follow_up": {"type": "ARRAY", "items": {"type": "STRING"}},
            "dietary_recommendations": {"type": "ARRAY", "items": {"type": "STRING"}},
            "physical_activity": {"type": "ARRAY", "items": {"type": "STRING"}},
            "mental_health_considerations": {"type": "ARRAY", "items": {"type": "STRING"}}
        },
        "required": ["medications", "lifestyle_modifications", "follow_up", "dietary_recommendations", "physical_activity", "mental_health_considerations"]
    }
    return await call_gemini_api(prompt, response_schema=schema)

async def generate_health_insights(health_metrics):
    metrics_str = "\n".join([f"- {k}: {v}" for k, v in health_metrics.items()])
    prompt = f"""
    Based on the following health metrics, provide key health insights and improvement recommendations.
    Focus on trends, potential concerns, and actionable advice.

    Health Metrics:
    {metrics_str}

    AI-Generated Health Insights and Recommendations:
    """
    return await call_gemini_api(prompt)

def generate_sample_health_data(days=90):
    data = []
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days - 1)

    for i in range(days):
        date = start_date + timedelta(days=i)
        heart_rate = random.randint(60, 100)
        blood_pressure_systolic = random.randint(110, 140)
        blood_pressure_diastolic = random.randint(70, 90)
        blood_glucose = random.randint(80, 120)
        
        if i > days * 0.7:
            heart_rate += random.randint(0, 10)
        if i > days * 0.5 and i < days * 0.6:
            blood_glucose += random.randint(10, 30)

        data.append({
            'Date': date.strftime('%Y-%m-%d'),
            'Heart Rate': heart_rate,
            'Blood Pressure Systolic': blood_pressure_systolic,
            'Blood Pressure Diastolic': blood_pressure_diastolic,
            'Blood Glucose': blood_glucose,
            'Symptom': random.choice(['None', 'Headache', 'Fatigue', 'Cough', 'Body Ache', 'Fever', 'Sore Throat'])
        })
    df = pd.DataFrame(data)

    symptom_counts = df[df['Symptom'] != 'None']['Symptom'].value_counts().reset_index()
    symptom_counts.columns = ['Symptom', 'Frequency']

    return df, symptom_counts

st.set_page_config(layout="wide", page_title="HealthAI - Intelligent Healthcare Assistant", page_icon="⚕")

st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6;
    }
    .main .block-container {
        padding-top: 1rem;
        padding-right: 1rem;
        padding-left: 1rem;
        padding-bottom: 1rem;
    }
    .sidebar .sidebar-content {
        background-color: #e0f2f1;
        border-radius: 10px;
        padding: 20px;
    }
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1.1rem;
        font-weight: bold;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 20px;
        font-size: 16px;
        box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #45a049;
        box-shadow: 0 8px 16px 0 rgba(0,0,0,0.2);
    }
    .chat-bubble-user {
        background-color: #e6e6e6;
        border-radius: 15px;
        padding: 10px 15px;
        margin: 5px 0;
        align-self: flex-end;
        max-width: 70%;
    }
    .chat-bubble-ai {
        background-color: #d1e7dd;
        border-radius: 15px;
        padding: 10px 15px;
        margin: 5px 0;
        align-self: flex-start;
        max-width: 70%;
    }
    .stMarkdown h3 {
        color: #00796b;
    }
    .stMetric {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .stPlotlyChart {
        border-radius: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        overflow: hidden;
    }
    .icon-large {
        font-size: 2em;
        margin-right: 10px;
    }
</style>
""", unsafe_allow_html=True)

if 'model_initialized' not in st.session_state:
    st.session_state['model_initialized'] = False
if 'patient_profile' not in st.session_state:
    st.session_state['patient_profile'] = {
        'name': 'Rithvik',
        'age': 22,
        'gender': 'Male',
        'medical_history': 'None',
        'current_medications': 'None',
        'allergies': 'None',
        'avg_heart_rate': 75,
        'avg_bp_systolic': 120,
        'avg_bp_diastolic': 80,
        'avg_glucose': 90,
        'recent_symptoms': 'None'
    }
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []
if 'health_data' not in st.session_state:
    st.session_state['health_data'], st.session_state['symptom_frequency'] = generate_sample_health_data()

st.sidebar.title("Patient Profile")
with st.sidebar:
    st.session_state['patient_profile']['name'] = st.text_input("Name", st.session_state['patient_profile']['name'])
    st.session_state['patient_profile']['age'] = st.number_input("Age", min_value=0, max_value=120, value=st.session_state['patient_profile']['age'])
    st.session_state['patient_profile']['gender'] = st.selectbox("Gender", ["Male", "Female", "Other"], index=["Male", "Female", "Other"].index(st.session_state['patient_profile']['gender']))
    st.session_state['patient_profile']['medical_history'] = st.text_area("Medical History", st.session_state['patient_profile']['medical_history'])
    st.session_state['patient_profile']['current_medications'] = st.text_area("Current Medications", st.session_state['patient_profile']['current_medications'])
    st.session_state['patient_profile']['allergies'] = st.text_area("Allergies", st.session_state['patient_profile']['allergies'])

    st.subheader("Recent Metrics (for AI context)")
    st.session_state['patient_profile']['avg_heart_rate'] = st.number_input("Avg. Heart Rate (bpm)", value=st.session_state['patient_profile']['avg_heart_rate'])
    col_bp1, col_bp2 = st.columns(2)
    with col_bp1:
        st.session_state['patient_profile']['avg_bp_systolic'] = st.number_input("Systolic BP", value=st.session_state['patient_profile']['avg_bp_systolic'])
    with col_bp2:
        st.session_state['patient_profile']['avg_bp_diastolic'] = st.number_input("Diastolic BP", value=st.session_state['patient_profile']['avg_bp_diastolic'])
    st.session_state['patient_profile']['avg_glucose'] = st.number_input("Avg. Blood Glucose (mg/dL)", value=st.session_state['patient_profile']['avg_glucose'])
    st.session_state['patient_profile']['recent_symptoms'] = st.text_area("Recently Reported Symptoms", st.session_state['patient_profile']['recent_symptoms'])

    st.markdown("---")
    st.info("💡 Patient Profile data is used by AI for personalized responses.")


st.title("⚕ HealthAI - Intelligent Healthcare Assistant")

if not st.session_state.model_initialized:
    init_granite_model()
    st.toast("HealthAI model initialized!")


tab1, tab2, tab3, tab4 = st.tabs(["💬 Patient Chat", "🩺 Disease Prediction", "💊 Treatment Plans", "📊 Health Analytics"])

with tab1:
    st.header("24/7 Patient Support")
    st.write("Ask any health-related question for immediate assistance.")

    chat_container = st.container(height=400)
    with chat_container:
        for entry in st.session_state['chat_history']:
            if entry["role"] == "user":
                st.markdown(f'<div class="chat-bubble-user">*You:* {entry["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-bubble-ai">*HealthAI:* {entry["content"]}</div>', unsafe_allow_html=True)

    user_query = st.text_input("Ask your health question...", key="chat_input", placeholder="e.g., What are the common symptoms of flu?")

    if user_query:
        st.session_state['chat_history'].append({"role": "user", "content": user_query})
        with st.spinner("HealthAI is thinking..."):
            ai_response = asyncio.run(answer_patient_query(user_query, st.session_state['chat_history']))
            st.session_state['chat_history'].append({"role": "ai", "content": ai_response})
        st.rerun()

with tab2:
    st.header("Disease Prediction System")
    st.write("Enter symptoms and patient data to receive potential condition predictions.")

    symptoms_input = st.text_area(
        "Describe symptoms in detail (e.g., persistent headache for 3 days, fatigue, mild fever of 99.5F)",
        st.session_state['patient_profile']['recent_symptoms'],
        height=100
    )

    if st.button("Generate Prediction", key="predict_btn"):
        if symptoms_input:
            with st.spinner("Analyzing symptoms and predicting potential conditions..."):
                prediction_output = asyncio.run(predict_disease(symptoms_input, st.session_state['patient_profile']))

            if prediction_output and prediction_output.get('potential_conditions'):
                st.subheader("Potential Conditions:")
                for i, condition in enumerate(prediction_output['potential_conditions']):
                    st.markdown(f"""
                    *{i+1}. {condition['name']}* (Likelihood: {condition['likelihood']})
                    Explanation: {condition['explanation']}
                    Next Steps: {condition['next_steps']}
                    """)
            else:
                st.warning("No clear predictions could be generated based on the provided symptoms. Please consult a medical professional.")
        else:
            st.warning("Please enter your symptoms to generate a prediction.")

with tab3:
    st.header("Personalized Treatment Plan Generator")
    st.write("Generate customized treatment recommendations based on specific conditions.")

    medical_condition = st.text_input("Medical Condition", placeholder="e.g., Mouth Ulcer")

    if st.button("Generate Treatment Plan", key="treatment_btn"):
        if medical_condition:
            with st.spinner("Generating personalized treatment plan..."):
                treatment_plan_output = asyncio.run(generate_treatment_plan(medical_condition, st.session_state['patient_profile']))

            if treatment_plan_output:
                st.subheader("Personalized Treatment Plan:")
                if treatment_plan_output.get("medications"):
                    st.markdown("*1. Recommended medications:*")
                    for med in treatment_plan_output["medications"]:
                        st.write(f"- {med}")
                if treatment_plan_output.get("lifestyle_modifications"):
                    st.markdown("*2. Lifestyle modifications:*")
                    for mod in treatment_plan_output["lifestyle_modifications"]:
                        st.write(f"- {mod}")
                if treatment_plan_output.get("follow_up"):
                    st.markdown("*3. Follow-up testing and monitoring:*")
                    for fu in treatment_plan_output["follow_up"]:
                        st.write(f"- {fu}")
                if treatment_plan_output.get("dietary_recommendations"):
                    st.markdown("*4. Dietary recommendations:*")
                    for diet in treatment_plan_output["dietary_recommendations"]:
                        st.write(f"- {diet}")
                if treatment_plan_output.get("physical_activity"):
                    st.markdown("*5. Physical activity guidelines:*")
                    for pa in treatment_plan_output["physical_activity"]:
                        st.write(f"- {pa}")
                if treatment_plan_output.get("mental_health_considerations"):
                    st.markdown("*6. Mental health considerations:*")
                    for mh in treatment_plan_output["mental_health_considerations"]:
                        st.write(f"- {mh}")
            else:
                st.warning("Could not generate a treatment plan. Please try again or provide more details.")
        else:
            st.warning("Please enter a medical condition to generate a treatment plan.")

with tab4:
    st.header("Health Analytics Dashboard")
    st.write("Visualize and analyze patient health data trends.")

    df_health = st.session_state['health_data']
    df_symptom_freq = st.session_state['symptom_frequency']

    st.subheader("Health Metrics Summary")
    col1, col2, col3, col4 = st.columns(4)

    latest_hr = df_health['Heart Rate'].iloc[-1]
    avg_hr = df_health['Heart Rate'].mean()
    hr_delta = latest_hr - avg_hr
    col1.metric("Avg. Heart Rate", f"{avg_hr:.0f} bpm", f"{hr_delta:.0f}")

    latest_bp_sys = df_health['Blood Pressure Systolic'].iloc[-1]
    latest_bp_dia = df_health['Blood Pressure Diastolic'].iloc[-1]
    avg_bp_sys = df_health['Blood Pressure Systolic'].mean()
    avg_bp_dia = df_health['Blood Pressure Diastolic'].mean()
    bp_sys_delta = latest_bp_sys - avg_bp_sys
    bp_dia_delta = latest_bp_dia - avg_bp_dia
    col2.metric("Avg. Blood Pressure", f"{avg_bp_sys:.0f}/{avg_bp_dia:.0f}", f"{bp_sys_delta:.0f}/{bp_dia_delta:.0f}")

    latest_glucose = df_health['Blood Glucose'].iloc[-1]
    avg_glucose = df_health['Blood Glucose'].mean()
    glucose_delta = latest_glucose - avg_glucose
    col3.metric("Avg. Blood Glucose", f"{avg_glucose:.1f} mg/dL", f"{glucose_delta:.1f}")

    avg_sleep = 7.2
    sleep_delta = -0.4
    col4.metric("Avg. Sleep", f"{avg_sleep:.1f} hours", f"{sleep_delta:.1f}")

    st.markdown("---")

    st.subheader("Health Trend Charts (90-Day)")

    fig_hr = px.line(df_health, x='Date', y='Heart Rate', title='Heart Rate Trend (90-Day)')
    fig_hr.update_traces(line_color='#2ecc71')
    st.plotly_chart(fig_hr, use_container_width=True)

    fig_bp = go.Figure()
    fig_bp.add_trace(go.Scatter(x=df_health['Date'], y=df_health['Blood Pressure Systolic'], mode='lines', name='Systolic'))
    fig_bp.add_trace(go.Scatter(x=df_health['Date'], y=df_health['Blood Pressure Diastolic'], mode='lines', name='Diastolic'))
    fig_bp.update_layout(title='Blood Pressure Trend (90-Day)', yaxis_title='Blood Pressure (mmHg)')
    st.plotly_chart(fig_bp, use_container_width=True)

    fig_glucose = px.line(df_health, x='Date', y='Blood Glucose', title='Blood Glucose Trend (90-Day)')
    fig_glucose.add_hline(y=99, line_dash="dot", line_color="red", annotation_text="Upper Normal Limit", annotation_position="top right")
    fig_glucose.update_traces(line_color='#e74c3c')
    st.plotly_chart(fig_glucose, use_container_width=True)

    if not df_symptom_freq.empty:
        fig_symptom = px.pie(df_symptom_freq, values='Frequency', names='Symptom', title='Symptom Frequency (90-Day)')
        fig_symptom.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_symptom, use_container_width=True)
    else:
        st.info("No reported symptoms to display frequency.")

    st.markdown("---")

    st.subheader("AI-Generated Health Insights")
    current_metrics = {
        "Avg. Heart Rate": f"{avg_hr:.0f} bpm",
        "Avg. Blood Pressure": f"{avg_bp_sys:.0f}/{avg_bp_dia:.0f} mmHg",
        "Avg. Blood Glucose": f"{avg_glucose:.1f} mg/dL",
        "Avg. Sleep": f"{avg_sleep:.1f} hours"
    }

    if st.button("Generate AI Insights", key="ai_insights_btn"):
        with st.spinner("Generating insights..."):
            insights = asyncio.run(generate_health_insights(current_metrics))
            st.markdown(insights)
            st.rerun()
    else:
        st.info("Click 'Generate AI Insights' to get AI-driven observations and recommendations based on current trends.")
        print("Code Successfully Executed")