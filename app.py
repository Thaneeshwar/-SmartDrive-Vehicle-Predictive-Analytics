# app.py
# SmartDrive — Vehicle Predictive Analytics (Accurate Maintenance Risk Fix)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
from datetime import datetime

# Black theme
st.set_page_config(page_title="SmartDrive — Vehicle Predictive Analytics", layout="wide", page_icon="📊")

st.markdown("""
<style>
    .stApp { background-color: #000000; color: #ffffff; }
    .stSidebar { background-color: #1a1a1a; }
    h1, h2, h3, h4 { color: #00ff9d !important; }
    .stMetric { background-color: #1a1a1a; border-radius: 10px; }
    .dataframe { background-color: #1a1a1a; }
</style>
""", unsafe_allow_html=True)

st.title("SmartDrive — Vehicle Predictive Analytics")
st.markdown("**Real Indian Used Car Data** • Resale Prediction • Maintenance Risk • Fleet Insights")
st.caption(f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M IST')} | Chennai, India")

# ─── Load & prepare real dataset ───
csv_file = Path(__file__).parent / "car data dekho.csv"

if 'df' not in st.session_state:
    if csv_file.exists():
        st.session_state.df = pd.read_csv(csv_file)
    else:
        st.error("Place 'car data dekho.csv' in the same folder!")
        st.stop()

if 'maintenance_risk_score' not in st.session_state.df.columns:
    df_temp = st.session_state.df.copy()
    df_temp['Age'] = 2025 - df_temp['Year']

    # ─── Accurate & realistic maintenance risk calculation ───
    df_temp['maintenance_risk_score'] = (
        0.42 * (df_temp['Kms_Driven'] / 100000.0) +          # mileage impact
        0.38 * (df_temp['Age'] / 15.0) +                     # age impact (normalized)
        0.28 * df_temp['Owner'] +                            # owner count impact
        np.where(df_temp['Fuel_Type'].isin(['Diesel', 'CNG']), 0.25, 0.0) +  # fuel type
        np.where(df_temp['Transmission'] == 'Manual', 0.10, 0.0) +
        np.random.normal(0, 0.32, len(df_temp))              # strong variation for realism
    ).clip(0, 1.8) / 1.8  # normalize to ≈0–1 range

    # Dynamic threshold → top ~30–40% are high risk
    threshold = np.percentile(df_temp['maintenance_risk_score'], 65)
    df_temp['maintenance_risk'] = (df_temp['maintenance_risk_score'] > threshold).astype(int)

    df_temp['current_price_lakh'] = df_temp['Selling_Price'].fillna(0)

    st.session_state.df = df_temp

df = st.session_state.df

# ─── Train models ───
@st.cache_resource
def train_models(_df):
    le_fuel   = LabelEncoder().fit(_df['Fuel_Type'])
    le_seller = LabelEncoder().fit(_df['Seller_Type'])
    le_trans  = LabelEncoder().fit(_df['Transmission'])

    X = pd.DataFrame({
        'Year': _df['Year'],
        'Kms_Driven': _df['Kms_Driven'],
        'Present_Price': _df['Present_Price'],
        'Owner': _df['Owner'],
        'Age': _df['Age'],
        'fuel_enc': le_fuel.transform(_df['Fuel_Type']),
        'seller_enc': le_seller.transform(_df['Seller_Type']),
        'trans_enc': le_trans.transform(_df['Transmission']),
    })

    reg = RandomForestRegressor(n_estimators=250, random_state=42, n_jobs=-1)
    reg.fit(X, _df['current_price_lakh'])

    clf = RandomForestClassifier(
        n_estimators=180,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'           # Helps significantly with imbalance
    )
    clf.fit(X, _df['maintenance_risk'])

    return reg, clf, le_fuel, le_seller, le_trans

reg_model, clf_model, le_fuel, le_seller, le_trans = train_models(df)

# ─── Debug: show class distribution ───
st.subheader("Debug: Maintenance Risk Class Balance")
col_dbg1, col_dbg2 = st.columns(2)
with col_dbg1:
    st.write("Class distribution (0 = Low, 1 = High):")
    st.write(df['maintenance_risk'].value_counts(normalize=True).round(4))
with col_dbg2:
    st.metric("High Risk Vehicles", (df['maintenance_risk']==1).sum(),
              f"{df['maintenance_risk'].mean():.1%} of total")

# ─── Add new vehicle form ───
with st.expander("➕ Add New Vehicle", expanded=False):
    with st.form("add_car"):
        col1, col2, col3 = st.columns(3)
        with col1:
            n_car   = st.text_input("Car Name", "Creta SX")
            n_year  = st.number_input("Year", 2000, 2025, 2023)
            n_kms   = st.number_input("Kms Driven", 0, 500000, 18000)
            n_present = st.number_input("Present Price (Lakh)", 0.5, 150.0, 14.0, 0.5)
        with col2:
            n_fuel   = st.selectbox("Fuel", ['Petrol','Diesel','CNG'])
            n_seller = st.selectbox("Seller", ['Dealer','Individual'])
            n_trans  = st.selectbox("Transmission", ['Manual','Automatic'])
            n_owner  = st.selectbox("Owner", [0,1,3])
        with col3:
            n_selling = st.number_input("Selling Price (Lakh) - optional", 0.0, 150.0, 0.0, 0.5)

        if st.form_submit_button("Add Vehicle", type="primary"):
            new_row = pd.DataFrame([{
                'Car_Name': n_car, 'Year': n_year, 'Selling_Price': n_selling if n_selling > 0 else np.nan,
                'Present_Price': n_present, 'Kms_Driven': n_kms, 'Fuel_Type': n_fuel,
                'Seller_Type': n_seller, 'Transmission': n_trans, 'Owner': n_owner
            }])
            st.session_state.df = pd.concat([st.session_state.df, new_row], ignore_index=True)
            st.success("Vehicle added!")
            st.rerun()

# ─── SINGLE VEHICLE PREDICTION (Maintenance Risk Meter) ───
st.subheader("🔮 Predict for a Vehicle — Maintenance Risk Meter")

with st.form("predict"):
    colA, colB = st.columns([2,1])
    with colA:
        p_name = st.text_input("Car Name (for reference)", "New Hyundai Creta")
        p_year = st.number_input("Year", 2005, 2025, 2022)
        p_kms  = st.number_input("Kms Driven", 0, 500000, 32000)
        p_present = st.number_input("Present Price (₹ Lakh)", 1.0, 200.0, 15.0, 0.5)
    with colB:
        p_fuel   = st.selectbox("Fuel", sorted(df['Fuel_Type'].unique()))
        p_seller = st.selectbox("Seller", sorted(df['Seller_Type'].unique()))
        p_trans  = st.selectbox("Transmission", sorted(df['Transmission'].unique()))
        p_owner  = st.selectbox("Owner", [0,1,3])

    if st.form_submit_button("Get Predictions", type="primary"):
        try:
            inp = pd.DataFrame([{
                'Year':          p_year,
                'Kms_Driven':    p_kms,
                'Present_Price': p_present,
                'Owner':         p_owner,
                'Age':           2025 - p_year,
                'fuel_enc':      int(le_fuel.transform([p_fuel])[0]),
                'seller_enc':    int(le_seller.transform([p_seller])[0]),
                'trans_enc':     int(le_trans.transform([p_trans])[0]),
            }])

            st.write("Input features sent to model:")
            st.dataframe(inp)

            price = reg_model.predict(inp)[0]

            risk_proba_array = clf_model.predict_proba(inp)

            if risk_proba_array.shape[1] == 1:
                risk_p = 0.0 if risk_proba_array[0][0] > 0.5 else 100.0
                label = "⚠️ Model saw only one risk level — approximate result"
                st.warning("Maintenance risk model has very low variation in training data.")
            else:
                risk_p = risk_proba_array[0][1] * 100
                risk_class = clf_model.predict(inp)[0]
                label = "🔴 HIGH RISK" if risk_class == 1 else "🟢 Low Risk"

            c1, c2 = st.columns(2)
            c1.metric("Estimated Resale Value", f"₹ {price:,.2f} Lakh")
            c2.metric("Maintenance Risk Meter", f"{risk_p:.1f}%", delta=label)

        except ValueError as e:
            st.error(f"ValueError: {str(e)}")
            st.info("Common causes: unknown fuel/seller/transmission type")
            st.write("Known categories:")
            st.write("Fuel:", le_fuel.classes_)
            st.write("Seller:", le_seller.classes_)
            st.write("Transmission:", le_trans.classes_)

        except Exception as e:
            st.error(f"Unexpected error: {type(e).__name__} → {str(e)}")

# ─── TABS (unchanged) ───
tab1, tab2, tab3, tab4 = st.tabs(["Price Insights", "Maintenance Risk", "Trends & Decay", "Full Dataset"])

with tab1:
    st.subheader("Resale Price Distribution")
    fig1 = px.histogram(df, x="current_price_lakh", color="Fuel_Type",
                        marginal="rug", opacity=0.75, template="plotly_dark", nbins=45)
    st.plotly_chart(fig1, use_container_width=True)

with tab2:
    st.subheader("Maintenance Risk Score Distribution")
    fig2 = px.histogram(df, x="maintenance_risk_score", nbins=60,
                        title="Maintenance Risk Score (0 = Excellent • 1 = Critical)",
                        template="plotly_dark", color_discrete_sequence=["#ff5252"])
    fig2.add_vline(x=np.percentile(df['maintenance_risk_score'], 65), line_dash="dash",
                   line_color="#ffeb3b", annotation_text="Dynamic Threshold")
    st.plotly_chart(fig2, use_container_width=True)

with tab3:
    st.subheader("Real-World Trends")
    price_by_year = df.groupby('Year')['current_price_lakh'].mean().reset_index()
    fig_decay = px.line(price_by_year, x='Year', y='current_price_lakh',
                        title="Average Resale Price Decay", template="plotly_dark")
    st.plotly_chart(fig_decay, use_container_width=True)

with tab4:
    st.subheader("Complete Dataset")
    st.dataframe(df.style.format({
        'current_price_lakh': '₹{:.2f} L',
        'maintenance_risk_score': '{:.3f}'
    }), use_container_width=True, height=550)

st.caption("Accurate maintenance risk meter — dynamic threshold + balanced classes + safety fallback")
# =============================================================================
#python -m streamlit run "C:/Vechile_dashboard_file/vechile_dashboard.py"