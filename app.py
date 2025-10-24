import streamlit as st

# -----------------------------
# 🎨 PAGE CONFIGURATION
# -----------------------------
st.set_page_config(page_title="Financial Health Assistant", layout="wide")

# -----------------------------
# 🌈 CUSTOM CSS FOR BACKGROUND + STYLE
# -----------------------------
def add_css():
    st.markdown("""
        <style>
        /* Gradient background (purple → black) */
        .stApp {
            background: linear-gradient(135deg, #3a0ca3, #000000);
            background-attachment: fixed;
            color: white;
            font-family: 'Poppins', sans-serif;
            overflow: hidden;
            animation: fadeInApp 1.2s ease-in-out forwards;
        }

        /* Container styling */
        .block-container {
            padding-top: 3rem;
            padding-bottom: 3rem;
            position: relative;
            z-index: 2;
        }

        /* Headings */
        h1, h2, h3, h4, h5, h6 {
            color: #ffffff !important;
            text-shadow: 0px 0px 10px rgba(186, 85, 211, 0.6);
        }

        /* Paragraphs and labels */
        p, label, span, div {
            color: #f1f1f1 !important;
        }

        /* Input fields */
        .stTextInput>div>div>input,
        .stNumberInput>div>div>input {
            background-color: rgba(255, 255, 255, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.3);
            color: white;
            border-radius: 10px;
        }

        /* Buttons */
        div.stButton>button {
            background: linear-gradient(90deg, #7209b7, #3a0ca3);
            color: white;
            border: none;
            border-radius: 12px;
            padding: 0.6rem 1.2rem;
            font-size: 1rem;
            font-weight: bold;
            box-shadow: 0 0 15px rgba(114, 9, 183, 0.6);
            transition: 0.3s ease-in-out;
        }
        div.stButton>button:hover {
            background: linear-gradient(90deg, #b5179e, #560bad);
            box-shadow: 0 0 25px rgba(181, 23, 158, 0.8);
            transform: scale(1.03);
        }

        /* Cards or results */
        .result-box {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 20px;
            padding: 20px;
            box-shadow: 0px 0px 20px rgba(0,0,0,0.3);
            backdrop-filter: blur(6px);
        }

        /* Fade animation */
        @keyframes fadeInApp {
            from {opacity: 0;}
            to {opacity: 1;}
        }
        </style>
    """, unsafe_allow_html=True)

add_css()

# -----------------------------
# 🧮 APP LOGIC
# -----------------------------
st.title("💰 Financial Health Assistant")
st.markdown("Analyze your financial health and get personalized recommendations.")

st.write("---")

col1, col2 = st.columns(2)

with col1:
    income = st.number_input("Monthly Income ($)", min_value=0.0, step=100.0)
    expenses = st.number_input("Monthly Expenses ($)", min_value=0.0, step=100.0)
    savings = st.number_input("Current Savings ($)", min_value=0.0, step=100.0)

with col2:
    debt = st.number_input("Total Debt ($)", min_value=0.0, step=100.0)
    goals = st.text_input("Financial Goal (e.g., Buy a house, Save for college)")
    risk = st.select_slider("Risk Tolerance", ["Low", "Moderate", "High"])

st.write("")
analyze = st.button("🔍 Analyze My Finances")

if analyze:
    st.markdown("<div class='result-box'>", unsafe_allow_html=True)

    if income == 0:
        st.error("Please enter a valid income to continue.")
    else:
        saving_rate = ((income - expenses) / income) * 100
        debt_ratio = (debt / income) * 100

        st.subheader("📊 Your Financial Overview")
        st.write(f"**Saving Rate:** {saving_rate:.2f}%")
        st.write(f"**Debt-to-Income Ratio:** {debt_ratio:.2f}%")

        if saving_rate > 20:
            st.success("✅ Excellent! You’re saving a healthy portion of your income.")
        elif 10 <= saving_rate <= 20:
            st.info("🟡 You’re doing okay — try to increase your savings slightly.")
        else:
            st.warning("🔴 You’re saving too little. Cut expenses or find ways to boost income.")

        if debt_ratio > 40:
            st.error("⚠️ High debt level! Try to reduce your liabilities.")
        else:
            st.success("💸 Your debt level is manageable.")

        st.write("---")
        st.subheader("🎯 Recommendation")
        if risk == "Low":
            st.write("→ Consider stable, low-risk investments like bonds or high-yield savings.")
        elif risk == "Moderate":
            st.write("→ A balanced portfolio of index funds and ETFs may suit you.")
        else:
            st.write("→ You could explore higher-risk, higher-return options like stocks or crypto (cautiously).")

        st.write(f"🎯 Financial Goal: *{goals if goals else 'No goal entered'}*")

    st.markdown("</div>", unsafe_allow_html=True)
