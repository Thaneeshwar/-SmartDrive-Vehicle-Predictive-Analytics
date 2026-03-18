# ─── SINGLE VEHICLE PREDICTION ───
st.subheader("🔮 Predict for a Vehicle")

with st.form("predict"):
    colA, colB = st.columns([2,1])
    with colA:
        p_name = st.text_input("Car Name", "New Hyundai Creta")
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
            # Prepare input in exactly the same order as training features
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

            # Optional: show what the model actually receives
            st.write("Input features being sent to model:")
            st.dataframe(inp)

            # 1. Resale value prediction
            price = reg_model.predict(inp)[0]

            # 2. Maintenance risk prediction with safety check
            risk_proba_array = clf_model.predict_proba(inp)

            if risk_proba_array.shape[1] == 1:
                # Fallback when model only learned one class
                risk_p = 0.0 if risk_proba_array[0][0] > 0.5 else 100.0
                label = "⚠️ Model imbalance - approximate result"
                st.warning("Maintenance risk model detected only one class in training data → result is approximate")
            else:
                risk_p = risk_proba_array[0][1] * 100   # probability of HIGH risk class
                risk_class = clf_model.predict(inp)[0]
                label = "🔴 HIGH RISK" if risk_class == 1 else "🟢 Low Risk"

            # Display results
            c1, c2 = st.columns(2)
            c1.metric("Estimated Resale Value", f"₹ {price:,.2f} Lakh")
            c2.metric("Maintenance Risk", f"{risk_p:.1f}%", delta=label)

        except ValueError as e:
            st.error(f"ValueError during prediction: {str(e)}")
            st.info("Common causes:\n"
                    "• Unknown category in Fuel/Seller/Transmission\n"
                    "• Mismatch in number/order of columns\n"
                    "• Data type issue (string instead of int/float)")
            st.write("Known categories:")
            st.write("Fuel:", le_fuel.classes_)
            st.write("Seller:", le_seller.classes_)
            st.write("Transmission:", le_trans.classes_)

        except Exception as e:
            st.error(f"Unexpected error: {type(e).__name__} → {str(e)}")
