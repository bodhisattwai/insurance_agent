# ======================================================================
# FILE: `main.py`
#
# INSTRUCTIONS:
# 1. This version uses the fast and reliable Groq API.
# 2. Replace the entire content of your `backend/main.py` file with this code.
# 3. You will add your new Groq API key as a secret on Render in the next step.
# ======================================================================

import joblib
import pandas as pd
import requests
from fastapi import FastAPI
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
import os

# --- Configuration ---
# --- NEW: Using Groq instead of Hugging Face ---
# This will read the API key from your hosting environment's secrets.
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLM_API_URL = "https://api.groq.com/openai/v1/chat/completions"
LLM_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct" # Groq is very fast with Llama 3

# --- FastAPI App Setup ---
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Load Local ML Models ---
try:
    scaler = joblib.load("scaler.pkl")
    model = joblib.load("kmeans_model.pkl")
    model_features = scaler.feature_names_in_
except FileNotFoundError:
    print("FATAL ERROR: 'scaler.pkl' or 'kmeans_model.pkl' not found.")
    scaler = None
    model = None
    model_features = []

# --- Pydantic Input Model ---
class ClientData(BaseModel):
    Age: int
    Income_Source: str
    annualIncome: str 
    familySize: int
    relationshipStatus: str
    financialGoal: str
    lifestyleFactors: list[str]

# --- Prediction Endpoint ---
@app.post("/predict")
def predict(data: ClientData):
    if not scaler or not model:
        return {"error": "Server is not ready. Models not loaded."}

    # --- Part 1: K-Means Persona Segmentation ---
    income_map = {"$15k-$30k": 22.5, "$30k-$50k": 40, "$50k-$75k": 62.5, "$75k-$100k": 87.5, "$100k+": 125}
    income_value = income_map.get(data.annualIncome, 0)

    input_data = {feature: 0 for feature in model_features}
    input_data['Age'] = data.Age
    input_data['Family Size'] = data.familySize
    input_data['Annual_Income_Value'] = income_value
    
    if f"Income Source_{data.Income_Source}" in input_data: input_data[f"Income Source_{data.Income_Source}"] = 1
    if f"Relationship Status_{data.relationshipStatus}" in input_data: input_data[f"Relationship Status_{data.relationshipStatus}"] = 1
    if f"Key Financial Goals_{data.financialGoal}" in input_data: input_data[f"Key Financial Goals_{data.financialGoal}"] = 1
    for factor in data.lifestyleFactors:
        if f"Lifestyle_{factor}" in input_data: input_data[f"Lifestyle_{factor}"] = 1
    
    input_df = pd.DataFrame([input_data], columns=model_features).apply(pd.to_numeric).fillna(0)

    try:
        scaled_features = scaler.transform(input_df)
        prediction = model.predict(scaled_features)[0]
        personas = { 0: "Ambitious Achiever", 1: "Pragmatic Protector", 2: "Established Guardian" }
        user_persona = personas.get(int(prediction), "Valued Client")
    except Exception as e:
        print(f"Error during K-Means prediction: {e}")
        user_persona = "Valued Client"

# --- Part 2: LLM-Powered Recommendation (IMPROVED) ---
llm_prompt = f"""
<role>
You are an expert, unbiased insurance advisor for Apeejay Insurance Broking, based in Kolkata, India. Your tone should be professional, clear, and encouraging.
</role>

<client_profile>
- Persona: {user_persona}
- Age: {data.Age}
- Income: {data.annualIncome} from {data.Income_Source}
- Family: {data.familySize} members, relationship status is {data.relationshipStatus}
- Primary Goal: {data.financialGoal}
- Lifestyle Factors: {', '.join(data.lifestyleFactors) or "None specified"}
</client_profile>

<task>
Generate a detailed, structured financial advisory report for the client. First, think step-by-step about their profile and needs. Then, write the report addressing them directly. Use real, well-known insurance products available in India. Do not invent product names.
</task>

<output_format>
The report must follow this exact markdown structure:

**1. Profile Summary:**
Briefly summarize your understanding of the client's current life stage and financial needs.

**2. Health Insurance Recommendation:**
Recommend the single most suitable health insurance plan from a top Indian provider (e.g., HDFC Ergo, Star Health, Niva Bupa). Justify your choice by connecting it directly to their profile (e.g., family size, age). Also, mention ONE strong alternative and briefly state why it's also a good option.

**3. Life Insurance Recommendation:**
Recommend the single most suitable type of life insurance policy (e.g., Term Plan, ULIP) from a top Indian provider (e.g., LIC, HDFC Life, ICICI Prudential). Justify why this policy type and provider are ideal for their primary financial goal.

**4. Important Considerations:**
Mention two key factors they should consider, such as specific riders (e.g., critical illness, accidental death) and the importance of checking the insurer's latest Claim Settlement Ratio (CSR).

**Disclaimer:**
End with a brief disclaimer stating that this is an AI-generated recommendation and a consultation with a human Apeejay advisor is recommended for personalized advice.
</output_format>
"""
    # --- NEW: Groq API Call Structure ---
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
            }
    payload = {
        "model": "llama4-scout", # Use the specific model name for Llama 4 Scout
        "messages": [{"role": "user", "content": llm_prompt}],
        "max_tokens": 800, # Increased slightly for the added alternative recommendation
        "temperature": 0.6, # Slightly lowered for more factual financial advice
            }

    try:
        print("Sending request to Groq LLM...")
        response = requests.post(LLM_API_URL, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        llm_result = response.json()['choices'][0]['message']['content']
        print("LLM response received successfully.")
        return {"name": user_persona, "recommendation": llm_result}
        
    except requests.exceptions.RequestException as e:
        print(f"\n--- LLM API ERROR ---")
        if hasattr(e, 'response') and e.response:
            print(f"Status Code: {e.response.status_code}")
            print(f"Response Body: {e.response.text}")
        else:
            print(f"An error occurred: {e}")
        print("--- END LLM API ERROR ---\n")
        
    return {
        "name": "Analysis Complete",
        "recommendation": "Based on your profile, it is highly recommended to secure both health and life insurance. A certified Apeejay advisor can help you select the best products from leading brands like HDFC, LIC, and Star Health to perfectly match your needs. Please contact us to discuss further."
    }
