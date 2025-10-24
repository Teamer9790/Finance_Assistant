import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
import plotly.express as px

# ---------------- PAGE CONFIG & CSS OVERHAUL ----------------
st.set_page_config(
    page_title="Pro Financial Assistant",
    page_icon="📈",
    layout="wide" # Use 'wide' layout for a true dashboard feel
)

def add_css():
    # Inject professional dark theme and custom component styling
    st.markdown("""
        <style>
        /* General dark theme setup */
        .stApp {
            background-color: #1e1e1e; /* Darker background */
            color: #f0f0f0;
        }
        /* Style for the main title */
        h1 {
            color: #00A3FF; /* Bright blue for emphasis */
            border-bottom: 2px solid #00A3FF;
            padding-bottom: 10px;
            margin-bottom: 20px;
        }
        /* Styling for subheaders */
        h2, h3 {
            color: #fafafa;
            border-left: 5px solid #00A3FF;
            padding-left: 10px;
            margin-top: 20px;
        }

        /* Custom Card Styles for Metrics (mimics Streamlit's containers but with custom look) */
        .metric-card {
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.4);
            transition: all 0.3s ease-in-out;
            margin-bottom: 15px;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }

        /* Specific Status Styling */
        .status-safe { background-color: #1a5c1a; } /* Dark Green */
        .status-moderate { background-color: #8f7220; } /* Dark Yellow */
        .status-critical { background-color: #7b2424; } /* Dark Red */

        /* Recommendations List Styling */
        .recommendation-list {
            list-style-type: none;
            padding: 0;
        }
        .recommendation-item {
            background-color: #2b2b2b; /* Slightly lighter dark color */
            padding: 10px 15px;
            margin-bottom: 8px;
            border-left: 4px solid #00A3FF;
            border-radius: 6px;
            font-size: 0.95rem;
            color: #d0d0d0;
        }
        .recommendation-item strong {
            color: #f0f0f0;
        }
        
        /* Custom Button Style */
        .stButton>button {
            width: 100%;
            background-color: #00A3FF;
            color: white;
            font-weight: bold;
            border-radius: 8px;
            border: none;
            padding: 10px;
            margin-top: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
            transition: background-color 0.2s;
        }
        .stButton>button:hover {
            background-color: #0089d0;
        }

        /* Sidebar Styling */
        .sidebar .stNumberInput {
            margin-bottom: 10px;
        }
        .sidebar .st-bb { /* Target the text inside sidebar */
            color: #d0d0d0;
        }
        
        /* Styling for Goal Planner result */
        .planner-advice {
            background-color: #2b2b2b;
            padding: 15px;
            border-radius: 8px;
            border: 2px solid #00A3FF;
        }

        </style>
        """, unsafe_allow_html=True)

# ---------------- DATA GENERATION ----------------
# No changes to backend data/model logic, only presentation.
def generate_data(n_samples=200):
    np.random.seed(42)
    data = {
        "income": np.random.randint(1_00_000, 1_00_00_000, n_samples),
        "side_income": np.random.randint(10_000, 20_00_000, n_samples),
        "annual_tax": np.random.randint(5_000, 20_00_000, n_samples),
        "loan": np.random.randint(50_000, 50_00_000, n_samples),
        "investment": np.random.randint(10_000, 30_00_000, n_samples),
        "personal_exp": np.random.randint(50_000, 50_00_000, n_samples),
        "emergency_exp": np.random.randint(10_000, 10_00_000, n_samples),
        "main_exp": np.random.randint(50_000, 30_00_000, n_samples)
    }
    df = pd.DataFrame(data)
    df["savings"] = (
        df["income"] + df["side_income"]
        - (df["annual_tax"] + df["loan"] + df["investment"] +
           df["personal_exp"] + df["emergency_exp"] + df["main_exp"])
    )
    df["status"] = np.where(
        (df["loan"] > df["income"] * 0.7) | (df["personal_exp"] > df["income"] * 0.8),
        "Critical",
        np.where(df["investment"] > df["income"] * 0.2, "Safe", "Moderate")
    )
    df["stability_score"] = np.random.randint(20, 100, n_samples)
    return df

# ---------------- MODEL TRAINING ----------------
# Using st.cache_resource for training models to prevent retraining on every interaction
@st.cache_resource
def train_models(_df):
    X = _df.drop(["status", "stability_score"], axis=1)
    y_class = _df["status"]
    y_reg = _df["stability_score"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    clf = RandomForestClassifier(random_state=42).fit(X_scaled, y_class)
    reg = RandomForestRegressor(random_state=42).fit(X_scaled, y_reg)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10).fit(X_scaled)
    return scaler, clf, reg, kmeans

# ---------------- RECOMMENDATIONS ----------------
def get_recommendations(values, result):
    income, side_income, annual_tax, loan, investment, personal_exp, emergency_exp, main_exp, savings = values
    recs = []
    
    # Debt Analysis
    debt_ratio = loan / (income + side_income) if (income + side_income) else 0
    if debt_ratio > 0.35:
        recs.append("⚠️ **High loan burden!** Your debt servicing ratio is high. Prioritize paying down high-interest debt or explore refinancing options.")
    else:
        recs.append("✅ **Loan levels are healthy.** Continue making timely payments and avoid taking on new, unnecessary debt.")
        
    # Investment Analysis
    investment_ratio = investment / income if income else 0
    if investment_ratio < 0.15:
        recs.append("📈 **Increase investment focus.** Aim to allocate at least 15-20% of your income towards growth investments for long-term compounding.")
    else:
        recs.append("✅ **Good investment ratio.** Keep diversifying your portfolio and consistently increasing your investment contributions.")

    # Emergency Fund Analysis (6 months of expenses, roughly 50% of income/year)
    if emergency_exp < (income * 0.1):
        recs.append("🚨 **Strengthen your emergency fund!** You need a more robust buffer. Aim for 3-6 months of living expenses saved in a liquid account.")
    else:
        recs.append("✅ **Emergency fund looks sufficient.** Ensure this fund remains separate from your main investments.")

    # Expense Control (50/30/20 Rule Check)
    total_needs = annual_tax + loan + emergency_exp + main_exp # Treating tax, loan, main exp as needs
    total_wants = personal_exp # Treating personal exp as wants
    
    needs_ratio = total_needs / (income + side_income) if (income + side_income) else 0
    wants_ratio = total_wants / (income + side_income) if (income + side_income) else 0

    if needs_ratio > 0.55:
        recs.append("💸 **Review your necessary spending.** Your essential living expenses are consuming too much of your income. Look for areas to reduce housing, transport, or insurance costs.")
    if wants_ratio > 0.3:
        recs.append("🛍️ **Control discretionary spending.** Personal expenses are high. Create a strict budget for 'wants' to free up capital for savings and investment.")

    # Savings check
    if savings < 0:
        recs.append("🔥 **IMMEDIATE ACTION REQUIRED:** You are overspending and incurring debt! Drastically cut all non-essential expenses immediately.")
    elif savings > (income + side_income) * 0.25:
        recs.append("🎯 **Excellent saving rate!** You are saving over 25% of your income. Continue this momentum.")
        
    return recs

# ---------------- ANALYSIS ----------------
def financial_assistant(values, scaler, clf, reg, kmeans):
    # Ensure 'savings' is the last item as it was in the training data
    if len(values) == 8: # If savings wasn't calculated yet
        savings = (values[0] + values[1]) - sum(values[2:8])
        values = values + (savings,)
    
    X_single_row = np.array(values).reshape(1, -1)
    
    # Scale all features except the last one (savings) which wasn't in the original X
    # NOTE: The training data (X) excluded 'status' and 'stability_score' but included 'savings'.
    # The input values array needs to match the structure of the training data 'X'.
    # Re-aligning input to match original 'X' structure:
    # X = df.drop(["status", "stability_score"], axis=1) -> [income, side_income, annual_tax, loan, investment, personal_exp, emergency_exp, main_exp, savings]
    
    # The input 'values' is already correctly structured [8 inputs + 1 calculated savings]
    
    # Drop the calculated savings value for scaling, since the original training X did not explicitly include it, but the values array has it.
    # Let's adjust the input to match the *original* column order: income through main_exp (8 features)
    input_features = np.array(values[:8]).reshape(1, -1)
    scaled = scaler.transform(input_features)
    
    status = clf.predict(scaled)[0]
    score = reg.predict(scaled)[0]
    cluster = kmeans.predict(scaled)[0]
    
    cluster_map = {0: "Conservative Saver", 1: "Aggressive Investor", 2: "Balanced Planner"}
    group = cluster_map.get(cluster, "Planner")
    
    if values[-1] < 0:
        group = "High Spender"
    
    result = {
        "Financial Status": status,
        "Stability Score": round(score[0] if isinstance(score, np.ndarray) else score, 2),
        "Group": group,
    }
    result["Recommendations"] = get_recommendations(values, result) # Pass full values for deep recs
    return result

# ---------------- SAVING PLANNER ----------------
def goal_saving_plan(goal_amount, months, total_income, total_expense):
    # Calculate monthly disposable income based on annual inputs
    disposable_income = (total_income - total_expense) / 12
    required_saving = goal_amount / months
    
    if disposable_income <= 0:
        status = "Impossible"
        advice = "Your annual expenses exceed your income. You must address overspending before planning goals."
    elif disposable_income >= required_saving:
        status = "Feasible"
        advice = "You're on track to meet your savings goal! Your current disposable income covers the required monthly saving."
    else:
        status = "Requires Adjustment"
        shortfall = required_saving - disposable_income
        advice = f"You need to increase your monthly disposable income by **₹{round(shortfall):,.0f}** or extend the time period to meet this goal."
        
    return {
        "Required Saving per Month": round(required_saving),
        "Disposable Income per Month": round(disposable_income),
        "Status": status,
        "Advice": advice
    }

# ---------------- MAIN ----------------
def main():
    add_css()
    df = generate_data()
    scaler, clf, reg, kmeans = train_models(df)

    st.title("💰 Financial Health Dashboard")
    st.markdown("Use the sidebar on the left to input your annual financial data (in ₹) and instantly analyze your financial health, stability, and future goals.")

    # ---------------- INPUT SIDEBAR ----------------
    with st.sidebar:
        st.header("Annual Financial Inputs (₹)")
        
        # Define default values and steps
        default_income = 12_00_000
        default_side_income = 2_00_000
        default_tax = 1_50_000
        default_loan = 4_00_000
        default_investment = 1_00_000
        default_personal_exp = 6_00_000
        default_emergency_exp = 80_000
        default_main_exp = 3_50_000
        
        income = st.number_input("Main Annual Income", min_value=0, value=default_income, step=10_000)
        side_income = st.number_input("Side Income", min_value=0, value=default_side_income, step=5_000)
        
        st.markdown("---")
        st.subheader("Expenses & Deductions")
        
        annual_tax = st.number_input("Annual Tax Paid", min_value=0, value=default_tax, step=5_000)
        loan = st.number_input("Yearly Loan Payments (EMI)", min_value=0, value=default_loan, step=10_000)
        investment = st.number_input("Total Annual Investments/Savings", min_value=0, value=default_investment, step=10_000)
        personal_exp = st.number_input("Personal/Discretionary Expenses", min_value=0, value=default_personal_exp, step=10_000)
        emergency_exp = st.number_input("Emergency Fund Contribution", min_value=0, value=default_emergency_exp, step=5_000)
        main_exp = st.number_input("Household/Essential Expenses", min_value=0, value=default_main_exp, step=10_000)

        total_income = income + side_income
        total_expense = annual_tax + loan + investment + personal_exp + emergency_exp + main_exp
        savings = total_income - total_expense

        st.markdown("---")
        
        # Placeholder for analysis button
        if st.button("📊 Run Financial Analysis", key="run_analysis"):
            st.session_state.run_analysis = True
        
        if "run_analysis" not in st.session_state:
             st.session_state.run_analysis = False


    # ---------------- MAIN DASHBOARD AREA ----------------
    
    if st.session_state.run_analysis:
        
        values = [income, side_income, annual_tax, loan, investment, personal_exp, emergency_exp, main_exp, savings]
        result = financial_assistant(values, scaler, clf, reg, kmeans)
        
        st.header("📈 Financial Summary & Key Metrics")
        
        # Color coding for status
        status_class_map = {"Safe": "status-safe", "Moderate": "status-moderate", "Critical": "status-critical"}
        status_emoji_map = {"Safe": "🟢", "Moderate": "🟡", "Critical": "🔴"}
        
        # --- 3 KEY METRIC CARDS ---
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(f"""
                <div class="metric-card {status_class_map.get(result['Financial Status'], 'status-moderate')}">
                    <p style="font-size: 1.1rem; margin: 0; color: #d0d0d0;">Financial Status</p>
                    <h3 style="margin: 5px 0 0 0; color: white;">{status_emoji_map.get(result['Financial Status'], '')} {result['Financial Status']}</h3>
                    <p style="font-size: 0.8rem; color: #f0f0f0;">Overall risk assessment</p>
                </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
                <div class="metric-card">
                    <p style="font-size: 1.1rem; margin: 0; color: #d0d0d0;">Stability Score</p>
                    <h3 style="margin: 5px 0 0 0; color: #00A3FF;">{result['Stability Score']}%</h3>
                    <p style="font-size: 0.8rem; color: #f0f0f0;">(Based on ML model prediction)</p>
                </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
                <div class="metric-card">
                    <p style="font-size: 1.1rem; margin: 0; color: #d0d0d0;">Savings / Overspend</p>
                    <h3 style="margin: 5px 0 0 0; color: {'#32CD32' if savings >= 0 else '#FF4500'};">₹{savings:,.0f}</h3>
                    <p style="font-size: 0.8rem; color: #f0f0f0;">({result['Group']} Group)</p>
                </div>
            """, unsafe_allow_html=True)
            
        st.markdown("---")

        # --- VISUAL BREAKDOWN & RECOMMENDATIONS ---
        col_chart, col_recs = st.columns([1.5, 1])
        
        with col_chart:
            st.subheader("📊 Annual Cash Flow Breakdown")
            
            # Prepare data for Plotly Pie Chart
            labels = ["Annual Tax", "Loan Payments", "Investment", "Personal Exp.", "Emergency Fund", "Household Exp."]
            sizes = [annual_tax, loan, investment, personal_exp, emergency_exp, main_exp]
            
            # Add savings/deficit to the chart
            if savings >= 0:
                labels.append("Net Savings")
                sizes.append(savings)
                colors = px.colors.qualitative.Prism + ['#32CD32'] # Add green for savings
            else:
                labels.append("Net Deficit")
                sizes.append(abs(savings))
                colors = px.colors.qualitative.Prism + ['#FF4500'] # Add red for deficit
            
            
            chart_df = pd.DataFrame({"Category": labels, "Amount": sizes})

            fig = px.pie(
                chart_df, 
                names='Category', 
                values='Amount', 
                hole=0.6, 
                color_discrete_sequence=colors,
                title=f"Total Outflow: ₹{total_expense:,.0f}"
            )
            
            # Update layout for dark theme
            fig.update_layout(
                paper_bgcolor="#1e1e1e", # Dashboard background color
                plot_bgcolor="#1e1e1e",
                font=dict(color="#f0f0f0", size=14),
                margin=dict(t=50, b=0, l=0, r=0),
                legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
            )
            fig.update_traces(textinfo='percent+label', marker=dict(line=dict(color='#1e1e1e', width=1)))
            
            st.plotly_chart(fig, use_container_width=True)

        with col_recs:
            st.subheader("💡 Actionable Recommendations")
            st.markdown("<ul class='recommendation-list'>", unsafe_allow_html=True)
            for rec in result["Recommendations"]:
                st.markdown(f"<li class='recommendation-item'>{rec}</li>", unsafe_allow_html=True)
            st.markdown("</ul>", unsafe_allow_html=True)


        st.markdown("---")
        
    # -------- Goal Saving Planner --------
    st.header("🎯 Financial Goal Planner")
    
    col_goal1, col_goal2 = st.columns(2)
    with col_goal1:
        goal_amount = st.number_input("Enter Goal Amount (₹)", min_value=0, value=5_00_000, step=10_000)
    with col_goal2:
        time_period = st.number_input("Enter Time Period (Months)", min_value=1, value=36, step=6)

    # Calculate plan based on current inputs (even if analysis wasn't run)
    plan = goal_saving_plan(goal_amount, time_period, total_income, total_expense)
    
    st.markdown(f"""
        <div class="planner-advice">
            <h4 style="color: #00A3FF; margin-top: 0;">Plan Status: {plan['Status']}</h4>
            <p><strong>Required Saving/Month:</strong> ₹{plan['Required Saving per Month']:,.0f}</p>
            <p><strong>Estimated Disposable Income/Month:</strong> ₹{plan['Disposable Income per Month']:,.0f}</p>
            <p style="font-weight: bold; margin-bottom: 0;">{plan['Advice']}</p>
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
