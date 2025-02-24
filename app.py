import streamlit as st
import joblib
import openai
import pandas as pd
import xgboost as xgb

# Load the trained model
model = joblib.load("XGB_model.pkl")

# Streamlit UI
st.set_page_config(page_title="HELOC Eligibility Predictor", page_icon="🏦", layout="wide")

# Sidebar for navigation
st.sidebar.image("https://via.placeholder.com/150", use_column_width=True)
st.sidebar.title("Navigation")
selection = st.sidebar.radio("Go to", ["Eligibility Check", "AI Chat Assistant"])

if selection == "Eligibility Check":
    st.title("🏠 HELOC Eligibility Predictor")
    st.write("🔍 Check your eligibility for a Home Equity Line of Credit.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 Applicant Information")
        external_risk_estimate = st.number_input("📊 Credit Score (0-100)", min_value=0, max_value=100, value=50)
        msince_recent_delq = st.number_input("📅 Months Since Last Delinquency", min_value=0, max_value=120, value=10)
        max_delq_ever = st.selectbox("⚠️ Max Delinquency Severity", options=list(range(9)), format_func=lambda x: f"Severity {x}" if x > 0 else "None")
    
    with col2:
        percent_trades_never_delq = st.number_input("📈 Percentage of Non-Delinquent Trades", min_value=0, max_value=100, value=80)
        msince_recent_inq = st.number_input("🔍 Months Since Last Credit Inquiry", min_value=0, max_value=100, value=10)
        api_key = st.text_input("🔑 Enter OpenAI API Key (Optional for AI Explanation)", type="password")
    
    feature_order = ["MSinceMostRecentDelq", "MaxDelqEver", "ExternalRiskEstimate", "PercentTradesNeverDelq", "MSinceMostRecentInqexcl7days"]
input_data = pd.DataFrame([[msince_recent_delq, max_delq_ever, external_risk_estimate, percent_trades_never_delq, msince_recent_inq]], columns=feature_order)
    
    if st.button("🚀 Check Eligibility"):
        try:
            probability = model.predict(input_data)[0]
            threshold = 0.5
            prediction = "✅ Eligible for Review" if probability >= threshold else "❌ Denied"
            st.subheader("📢 Prediction Result")
            st.markdown(f"### {prediction}")
            st.progress(probability)
            
            if prediction == "❌ Denied":
                explanation_text = (f"Your application was denied due to a low credit score ({external_risk_estimate}), "
                                    f"recent delinquencies ({msince_recent_delq} months ago), or max delinquency severity ({max_delq_ever}). "
                                    "Consider improving these factors before reapplying.")
                st.warning(explanation_text)
            
            if api_key:
                client = openai.OpenAI(api_key=api_key)
                try:
                    response = client.chat.completions.create(
                        model="gpt-4",
                        messages=[{"role": "user", "content": explanation_text}]
                    )
                    st.info("💡 AI Explanation:")
                    st.write(response.choices[0].message.content)
                except Exception as e:
                    st.error(f"⚠️ OpenAI API Error: {str(e)}")
        except Exception as e:
            st.error(f"⚠️ Model Prediction Error: {str(e)}")

elif selection == "AI Chat Assistant":
    st.title("💬 AI Chat Assistant")
    st.write("🤖 Ask AI any questions about your HELOC eligibility.")
    api_key = st.text_input("🔑 Enter OpenAI API Key", type="password")
    user_question = st.text_area("❓ Enter your question:")
    
    if st.button("🗣️ Get AI Response") and api_key:
        try:
            client = openai.OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": user_question}]
            )
            st.success("💬 AI Response:")
            st.write(response.choices[0].message.content)
        except Exception as e:
            st.error(f"⚠️ OpenAI API Error: {str(e)}")
