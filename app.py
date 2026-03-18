# app.py
# SmartDrive — Vehicle Predictive Analytics

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
from datetime import datetime

# ─── Page config & Black theme ───
st.set_page_config(
    page_title="SmartDrive — Vehicle Predictive Analytics",
    layout="wide",
    page_icon="📊"
)
st.markdown("""
<style>
    .stApp { background-color: #000000; color: #ffffff; }
    .stSidebar { background-color: #1a1a1a; }
    h1, h2, h3, h4 { color: #00ff9d !important; }
    .stMetric { background-color: #1a1a1a; border-radius: 10px; }
    .dataframe { background-color: #1a1a1a; }
    div[data-testid="stForm"] {
        background-color: #111111;
        border: 1px solid #00ff9d33;
        border-radius: 10px;
        padding: 16px;
    }
</style>
""", unsafe_allow_html=True)

st.title("SmartDrive — Vehicle Predictive Analytics")
st.markdown("**Real Indian Used Car Data** • Resale Prediction • Maintenance Risk • Fleet Insights")
st.caption(f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M IST')} | Chennai, India")


# ─── Helper: clean price columns ───
def clean_price_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    for col in ['Price', 'New_Price']:
        if col in dataframe.columns:
            dataframe[col] = (
                dataframe[col]
                .astype(str)
                .str.replace(r'[^\d.]', '', regex=True)
                .replace('', np.nan)
            )
            dataframe[col] = pd.to_numeric(dataframe[col], errors='coerce')
    return dataframe


# ─── Helper: compute risk score deterministically for a single vehicle ───
def compute_risk_score(km, age, owner_type, fuel_type, transmission):
    score = (
        0.42 * (km / 100_000.0)
        + 0.38 * (age / 15.0)
        + (0.00 if owner_type == 'First'
           else 0.25 if owner_type == 'Second'
           else 0.45 if owner_type == 'Third'
           else 0.60)
        + (0.25 if fuel_type in ['Diesel', 'CNG', 'LPG'] else 0.0)
        + (0.10 if transmission == 'Manual' else 0.0)
    )
    return float(np.clip(score / 1.8, 0, 1))


# ─── Load & prepare dataset ───
csv_file = Path(__file__).parent / "used_cars_data.csv"

if 'df' not in st.session_state:
    if csv_file.exists():
        raw_df = pd.read_csv(csv_file)
        raw_df = clean_price_columns(raw_df)
        raw_df['Price'] = raw_df['Price'].fillna(0)
        if 'New_Price' in raw_df.columns:
            raw_df['New_Price'] = raw_df['New_Price'].fillna(0)
        if 'Kilometers_Driven' in raw_df.columns:
            raw_df['Kilometers_Driven'] = pd.to_numeric(
                raw_df['Kilometers_Driven'], errors='coerce'
            )
        st.session_state.df = raw_df
    else:
        st.error("Place 'used_cars_data.csv' in the same folder as app.py!")
        st.stop()

# ─── Add derived columns once ───
if 'maintenance_risk_score' not in st.session_state.df.columns:
    df_temp = st.session_state.df.copy()
    df_temp['Age'] = 2025 - df_temp['Year']
    df_temp['maintenance_risk_score'] = (
        0.42 * (df_temp['Kilometers_Driven'].fillna(0) / 100_000.0)
        + 0.38 * (df_temp['Age'] / 15.0)
        + np.where(df_temp['Owner_Type'] == 'First',  0.00,
          np.where(df_temp['Owner_Type'] == 'Second', 0.25,
          np.where(df_temp['Owner_Type'] == 'Third',  0.45, 0.60)))
        + np.where(df_temp['Fuel_Type'].isin(['Diesel', 'CNG', 'LPG']), 0.25, 0.0)
        + np.where(df_temp['Transmission'] == 'Manual', 0.10, 0.0)
        + np.random.normal(0, 0.32, len(df_temp))
    ).clip(0, 1.8) / 1.8

    threshold = np.percentile(df_temp['maintenance_risk_score'], 65)
    df_temp['maintenance_risk'] = (df_temp['maintenance_risk_score'] > threshold).astype(int)
    df_temp['current_price_lakh'] = df_temp['Price']
    st.session_state.df = df_temp

df = st.session_state.df

# ─── Train models (cached) ───
@st.cache_resource
def train_models(dataframe: pd.DataFrame):
    le_dict = {}
    model_df = dataframe.copy()
    cat_cols = ['Name', 'Location', 'Fuel_Type', 'Transmission', 'Owner_Type']
    for col in cat_cols:
        if col in model_df.columns:
            le = LabelEncoder()
            model_df[col] = le.fit_transform(model_df[col].astype(str))
            le_dict[col] = le

    feature_cols = ['Year', 'Kilometers_Driven', 'Fuel_Type',
                    'Transmission', 'Owner_Type', 'Seats']
    feature_cols = [c for c in feature_cols if c in model_df.columns]

    X = model_df[feature_cols].fillna(0)
    y_price = model_df['Price'].fillna(0)
    y_risk  = model_df['maintenance_risk'].fillna(0).astype(int)

    price_model = RandomForestRegressor(n_estimators=100, random_state=42)
    price_model.fit(X, y_price)
    risk_model = RandomForestClassifier(n_estimators=100, random_state=42)
    risk_model.fit(X, y_risk)

    return price_model, risk_model, le_dict, feature_cols

price_model, risk_model, le_dict, feature_cols = train_models(df)


# ════════════════════════════════════════════════
# SIDEBAR — Predict a Vehicle  (with Car/Model Name)
# ════════════════════════════════════════════════
with st.sidebar:
    st.header("🚗 Predict a Vehicle")

    with st.form("vehicle_form"):
        st.markdown("##### 🏷️ Car / Model Name")
        car_name = st.text_input(
            "Name",
            placeholder="e.g. Maruti Swift VXi, Honda City ZX",
            label_visibility="collapsed"
        )

        st.markdown("##### ⚙️ Specifications")
        year  = st.number_input("Year",              min_value=2000, max_value=2025, value=2018)
        km    = st.number_input("Kilometers Driven", min_value=0,    max_value=500_000,
                                value=50_000, step=1000)
        fuel  = st.selectbox("Fuel Type",       ["Petrol", "Diesel", "CNG", "LPG", "Electric"])
        trans = st.selectbox("Transmission",    ["Manual", "Automatic"])
        owner = st.selectbox("Owner Type",      ["First", "Second", "Third", "Fourth & Above"])
        seats = st.number_input("Seats", min_value=2, max_value=10, value=5)

        submitted = st.form_submit_button("🔍 Predict", use_container_width=True)

    if submitted:
        input_data = {
            'Year': year, 'Kilometers_Driven': km,
            'Fuel_Type': fuel, 'Transmission': trans,
            'Owner_Type': owner, 'Seats': seats,
        }
        row = pd.DataFrame([input_data])
        for col in ['Fuel_Type', 'Transmission', 'Owner_Type']:
            if col in le_dict:
                known = list(le_dict[col].classes_)
                row[col] = row[col].apply(lambda v: v if v in known else known[0])
                row[col] = le_dict[col].transform(row[col])

        X_input    = row[feature_cols].fillna(0)
        pred_price = price_model.predict(X_input)[0]
        pred_risk  = risk_model.predict(X_input)[0]
        risk_prob  = risk_model.predict_proba(X_input)[0][1]

        display_name = car_name.strip() if car_name.strip() else "This Vehicle"
        st.markdown(f"### 📋 {display_name}")
        st.success(f"💰 Predicted Price: ₹{pred_price:,.2f} L")
        risk_label = "⚠️ High Risk" if pred_risk == 1 else "✅ Low Risk"
        st.info(f"🔧 Maintenance: {risk_label} ({risk_prob:.1%})")
        st.caption(f"Age: {2025 - year} yrs  |  {km:,} km driven")


# ─── Summary metrics ───
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Vehicles",     f"{len(df):,}")
col2.metric("Avg Price (₹ Lakh)", f"₹{df['Price'].mean():,.2f}")
col3.metric("High Risk Vehicles", f"{df['maintenance_risk'].sum():,}")
col4.metric("Cities Covered",     df['Location'].nunique() if 'Location' in df.columns else "N/A")

st.divider()

# ════════════════════════════════════════════════
# TABS
# ════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Price Distribution", "🔧 Risk Score", "📈 Trends", "📋 Dataset"
])

# ── Tab 1 ──
with tab1:
    st.subheader("Price Distribution by Fuel Type")
    fig1 = px.histogram(df, x="Price", color="Fuel_Type", template="plotly_dark",
                        labels={"Price": "Price (₹ Lakh)"}, title="Vehicle Price Distribution")
    st.plotly_chart(fig1, use_container_width=True)

# ── Tab 2 ──
with tab2:
    st.subheader("Maintenance Risk Score Distribution")
    fig2 = px.histogram(df, x="maintenance_risk_score", nbins=60, template="plotly_dark",
                        color_discrete_sequence=["#ff5252"],
                        labels={"maintenance_risk_score": "Risk Score"})
    st.plotly_chart(fig2, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        risk_counts = df['maintenance_risk'].value_counts().rename({0: "Low Risk", 1: "High Risk"})
        fig2b = px.pie(values=risk_counts.values, names=risk_counts.index,
                       template="plotly_dark",
                       color_discrete_sequence=["#00ff9d", "#ff5252"],
                       title="Risk Category Split")
        st.plotly_chart(fig2b, use_container_width=True)
    with c2:
        fig2c = px.box(df, x="maintenance_risk", y="Price", template="plotly_dark",
                       labels={"maintenance_risk": "Risk (0=Low, 1=High)", "Price": "Price (₹ Lakh)"},
                       title="Price vs Maintenance Risk")
        st.plotly_chart(fig2c, use_container_width=True)

# ── Tab 3 ──
with tab3:
    st.subheader("Price vs Year Trend")
    price_year = df.groupby('Year')['Price'].mean().reset_index()
    fig3 = px.line(price_year, x='Year', y='Price', template="plotly_dark",
                   labels={"Price": "Avg Price (₹ Lakh)"},
                   title="Average Resale Price Over Years")
    st.plotly_chart(fig3, use_container_width=True)

    if 'Location' in df.columns:
        st.subheader("Average Price by City")
        city_price = (df.groupby('Location')['Price'].mean()
                        .sort_values(ascending=False).reset_index())
        fig3b = px.bar(city_price, x='Location', y='Price', template="plotly_dark",
                       labels={"Price": "Avg Price (₹ Lakh)"},
                       title="City-wise Average Resale Price",
                       color='Price', color_continuous_scale='teal')
        st.plotly_chart(fig3b, use_container_width=True)


# ════════════════════════════════════════════════
# TAB 4 — Dataset  +  Add New Vehicle
# ════════════════════════════════════════════════
with tab4:

    # ── Section A: View Dataset ──
    st.subheader("📋 Complete Dataset")

    format_dict = {}
    if 'Price' in df.columns:
        format_dict['Price']     = lambda x: f"₹{x:,.2f} L" if pd.notna(x) and x != 0 else "-"
    if 'New_Price' in df.columns:
        format_dict['New_Price'] = lambda x: f"₹{x:,.2f} L" if pd.notna(x) and x != 0 else "-"
    if 'maintenance_risk_score' in df.columns:
        format_dict['maintenance_risk_score'] = '{:.3f}'

    st.dataframe(
        df.style.format(format_dict, na_rep="-"),
        use_container_width=True,
        height=420
    )

    csv_bytes = df.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download Dataset as CSV", data=csv_bytes,
                       file_name="smartdrive_cleaned_data.csv", mime="text/csv")

    st.divider()

    # ════════════════════════════════════════════
    # Section B: ➕ Add New Vehicle to Dataset
    # ════════════════════════════════════════════
    st.subheader("➕ Add New Vehicle to Dataset")
    st.markdown(
        "Fill in the fields below. **S.No**, **Age**, and **Maintenance Risk Score** "
        "are calculated automatically."
    )

    with st.form("add_vehicle_form", clear_on_submit=True):

        # Row 1 — Identity
        st.markdown("#### 🏷️ Identity")
        r1c1, r1c2 = st.columns(2)
        with r1c1:
            av_name     = st.text_input("Car / Model Name *",
                                        placeholder="e.g. Maruti Swift VXi")
        with r1c2:
            av_location = st.text_input("Location / City *",
                                        placeholder="e.g. Chennai, Mumbai")

        # Row 2 — Year, KM, Seats
        st.markdown("#### 📅 Year & Usage")
        r2c1, r2c2, r2c3 = st.columns(3)
        with r2c1:
            av_year  = st.number_input("Year *", min_value=2000, max_value=2025, value=2020)
        with r2c2:
            av_km    = st.number_input("Kilometers Driven *",
                                       min_value=0, max_value=500_000,
                                       value=30_000, step=1000)
        with r2c3:
            av_seats = st.number_input("Seats *", min_value=2, max_value=10, value=5)

        # Row 3 — Fuel / Transmission / Owner
        st.markdown("#### ⚙️ Classification")
        r3c1, r3c2, r3c3 = st.columns(3)
        with r3c1:
            av_fuel  = st.selectbox("Fuel Type *",
                                    ["Petrol", "Diesel", "CNG", "LPG", "Electric"])
        with r3c2:
            av_trans = st.selectbox("Transmission *", ["Manual", "Automatic"])
        with r3c3:
            av_owner = st.selectbox("Owner Type *",
                                    ["First", "Second", "Third", "Fourth & Above"])

        # Row 4 — Mileage / Engine / Power
        st.markdown("#### 🔧 Engine & Performance")
        r4c1, r4c2, r4c3 = st.columns(3)
        with r4c1:
            av_mileage = st.number_input("Mileage (kmpl)",
                                         min_value=0.0, max_value=80.0,
                                         value=18.0, step=0.1, format="%.1f")
        with r4c2:
            av_engine  = st.number_input("Engine (CC)",
                                          min_value=0, max_value=6000,
                                          value=1200, step=50)
        with r4c3:
            av_power   = st.number_input("Power (bhp)",
                                          min_value=0.0, max_value=600.0,
                                          value=82.0, step=1.0, format="%.1f")

        # Row 5 — Prices
        st.markdown("#### 💰 Pricing")
        r5c1, r5c2 = st.columns(2)
        with r5c1:
            av_price     = st.number_input("Resale Price (₹ Lakh) *",
                                            min_value=0.0, max_value=200.0,
                                            value=5.0, step=0.1, format="%.2f")
        with r5c2:
            av_new_price = st.number_input("New Price (₹ Lakh)  [optional]",
                                            min_value=0.0, max_value=500.0,
                                            value=0.0, step=0.1, format="%.2f")

        # Auto-computed preview (read-only info)
        _age_preview   = 2025 - av_year
        _risk_preview  = compute_risk_score(av_km, _age_preview, av_owner, av_fuel, av_trans)
        _sno_preview   = len(st.session_state.df) + 1

        st.markdown("#### 🤖 Auto-Computed Fields *(preview)*")
        ac1, ac2, ac3 = st.columns(3)
        ac1.info(f"**S.No:** {_sno_preview}")
        ac2.info(f"**Age:** {_age_preview} years")
        ac3.info(f"**Risk Score:** {_risk_preview:.3f}")

        add_submitted = st.form_submit_button("✅ Add Vehicle to Dataset",
                                              use_container_width=True)

    # ── Handle submission ──
    if add_submitted:
        if not av_name.strip():
            st.error("❌ Car / Model Name is required.")
        elif not av_location.strip():
            st.error("❌ Location / City is required.")
        elif av_price <= 0:
            st.error("❌ Resale Price must be greater than 0.")
        else:
            current_df    = st.session_state.df
            av_age        = 2025 - av_year
            av_risk_score = compute_risk_score(av_km, av_age, av_owner, av_fuel, av_trans)
            threshold_val = np.percentile(current_df['maintenance_risk_score'], 65)
            av_risk_label = int(av_risk_score > threshold_val)
            av_sno        = len(current_df) + 1

            new_row = {
                'S.No':                   av_sno,
                'Name':                   av_name.strip(),
                'Location':               av_location.strip(),
                'Year':                   av_year,
                'Kilometers_Driven':      av_km,
                'Fuel_Type':              av_fuel,
                'Transmission':           av_trans,
                'Owner_Type':             av_owner,
                'Mileage':                av_mileage,
                'Engine':                 av_engine,
                'Power':                  av_power,
                'Seats':                  av_seats,
                'New_Price':              av_new_price if av_new_price > 0 else np.nan,
                'Price':                  av_price,
                'Age':                    av_age,
                'maintenance_risk_score': av_risk_score,
                'maintenance_risk':       av_risk_label,
                'current_price_lakh':     av_price,
            }

            new_row_df = pd.DataFrame([new_row])

            # Align to existing columns — fill any extras with NaN
            for col in current_df.columns:
                if col not in new_row_df.columns:
                    new_row_df[col] = np.nan

            updated_df = pd.concat(
                [current_df, new_row_df[current_df.columns]],
                ignore_index=True
            )
            st.session_state.df = updated_df

            risk_str = "⚠️ High Risk" if av_risk_label == 1 else "✅ Low Risk"
            st.success(
                f"✅ **{av_name.strip()}** added successfully! "
                f"S.No: {av_sno} | Age: {av_age} yrs | "
                f"Risk Score: {av_risk_score:.3f} → {risk_str}"
            )

            # Preview the new row in a clean table
            st.markdown("##### 🆕 Newly Added Record")
            preview_cols = [
                'S.No', 'Name', 'Location', 'Year', 'Kilometers_Driven',
                'Fuel_Type', 'Transmission', 'Owner_Type',
                'Mileage', 'Engine', 'Power', 'Seats',
                'New_Price', 'Price', 'Age',
                'maintenance_risk_score', 'maintenance_risk'
            ]
            preview_cols = [c for c in preview_cols if c in updated_df.columns]
            st.dataframe(
                updated_df[preview_cols].tail(1).style.format({
                    'Price':                  lambda x: f"₹{x:,.2f} L" if pd.notna(x) and x != 0 else "-",
                    'New_Price':              lambda x: f"₹{x:,.2f} L" if pd.notna(x) and x != 0 else "-",
                    'maintenance_risk_score': '{:.3f}',
                }, na_rep="-"),
                use_container_width=True
            )

            st.rerun()

st.caption("SmartDrive | Prices in ₹ Lakh | Data cleaned & formatted safely")
#python -m streamlit run "C:/Vechile_dashboard_file/app.py"