import streamlit as st
from groq import Groq
from dotenv import load_dotenv
import os
import joblib

load_dotenv()

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="AI Smart Agriculture",
    page_icon="🌱",
    layout="wide"
)

# --------------------------------------------------
# MAIN TITLE
# --------------------------------------------------

st.title("🌱 AI-Based Smart Agriculture System")

st.write(
    "An AI-powered platform for crop recommendation, "
    "plant disease detection, smart irrigation and fertilizer recommendation."
)

st.success("Streamlit application started successfully!")
# Load crop recommendation AI model
crop_model = joblib.load("models/crop_model.pkl")

# --------------------------------------------------
# AGRICULTURE AI MODULES
# --------------------------------------------------
st.subheader("🌱 AI Crop Recommendation")

N = st.number_input("Nitrogen (N)", min_value=0.0)
P = st.number_input("Phosphorus (P)", min_value=0.0)
K = st.number_input("Potassium (K)", min_value=0.0)

temperature = st.number_input("Temperature (°C)")
humidity = st.number_input("Humidity (%)")
ph = st.number_input("Soil pH")
rainfall = st.number_input("Rainfall (mm)")

if st.button("🌱 Recommend Crop"):
    prediction = crop_model.predict([[
        N, P, K,
        temperature,
        humidity,
        ph,
        rainfall
    ]])[0]

    st.success(f"Recommended Crop: **{prediction}**")
st.subheader("🌾 Agriculture AI Modules")

col1, col2 = st.columns(2)

with col1:

    st.info("🌱 AI Crop Recommendation")

    st.write(
        "Recommend the best crop using soil and weather conditions."
    )

    st.info("🍃 AI Plant Disease Detection")

    st.write(
        "Detect plant diseases from leaf images."
    )


with col2:

    st.info("💧 AI Smart Irrigation")

    st.write(
        "Recommend irrigation based on environmental conditions."
    )

    st.info("🧪 AI Fertilizer Recommendation")

    st.write(
        "Recommend suitable fertilizer based on soil conditions."
    )

# --------------------------------------------------
# AGRICULTURE AI CHAT
# --------------------------------------------------

st.subheader("💬 Ask Agriculture AI")

question = st.text_input(
    "Ask your agriculture question:",
    placeholder="Example: Which crop is suitable for high rainfall?"
)

if question:

    try:

        # Connect to Groq
        client = Groq(
            api_key=os.getenv("GROQ_API_KEY")
        )


        # Send question to AI
        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an AI Agriculture Assistant. "
                        "Answer agriculture-related questions clearly "
                        "and simply. You can answer questions about "
                        "crops, soil, irrigation, fertilizers, "
                        "plant diseases and farming."
                    )
                },
                {
                    "role": "user",
                    "content": question
                }
            ]
        )

        # Get AI answer
        answer = response.choices[0].message.content

        st.success("🌱 AI Answer")

        st.write(answer)

    except Exception as e:

        st.error("AI connection error.")

        st.write(e)