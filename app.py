import streamlit as st
import pandas as pd
import joblib
import shap

background_data=joblib.load("models/background_data.pkl")

model=joblib.load("models/churn_model.pkl")
explainer=shap.Explainer(
    model,background_data
)
feature_names=joblib.load("models/feature_names.pkl")
scaler=joblib.load("models/scaler.pkl")

st.title("Customer_Churn_Prediction")
st.write("Enter Customer details below: ")

tenure=st.number_input(
    "Tenure (months)",
    min_value=0,
    max_value=72,
    value=12
)
monthly_charges=st.number_input(
    "Monthly Charges",
    min_value=0.0,
    value=50.0
)

gender=st.selectbox(
    "Gender",
    ["Male","Female"]
)
partner=st.selectbox(
    "Partner",
    ["Yes","No"]
)
senior_citizen=st.selectbox(
    "Senior Citizen",
    ["Yes","No"]
)
dependents=st.selectbox(
    "Dependents",
    ["Yes","No"]
)
paperless_billing=st.selectbox(
    "Paperless Billing",
    ["Yes","No"]
)
phone_service=st.selectbox(
    "Phone Service",
    ["Yes","No"]
)
multiple_lines=st.selectbox(
    "Multilple Lines",
    ["Yes","No","No phone service"]
)
internet_service=st.selectbox(
    "Internet Service",
    ["DSL","Fiber optic","No"]
)
if internet_service!="No":
    online_security=st.selectbox(
        "Online Security",
        ["Yes","No","No internet service"]
        )
    online_backup=st.selectbox(
        "Online Backup",
        ["Yes","No","No internet service"]
        )
    device_protection=st.selectbox(
        "Device protection",
        ["Yes","No","No internet service"]
        )
    tech_support=st.selectbox(
        "Tech Support",
        ["Yes","No","No internet service"]
        )
    streaming_tv=st.selectbox(
        "Streaming TV",
        ["Yes","No","No internet service"]
        )
    streaming_movies=st.selectbox(
        "Streaming Movies",
        ["Yes","No","No internet service"]
        )
contract=st.selectbox(
    "Contract",
    ["month-to-month","One year","Two year"]
)
payment_method=st.selectbox(
    "Payment Method",
    ["Electronic check","Mailed check","Bank transfer (automatic)","Credit card (automatic)"]
)
input_data={col: 0 for col in feature_names}
input_data["tenure"]=tenure
input_data["MonthlyCharges"]=monthly_charges
input_data["TotalCharges"]=tenure*monthly_charges

if gender=="Female":
    input_data["gender"]=1
if partner=="Yes":
    input_data["Partner"]=1
if dependents=="Yes":
    input_data["Dependents"]=1
if phone_service=="Yes":
    input_data["PhoneService"]=1
if paperless_billing=="Yes":
    input_data["PaperlessBilling"]=1
if senior_citizen=="Yes":
    input_data["SeniorCitizen"]=1

if contract=="One year":
    input_data["Contract_One year"]=1
elif contract=="Two year":
    input_data["Contract_Two year"]=1
if payment_method=="Credit card (automatic)":
    input_data["PaymentMethod_Credit card (automatic)"]=1
elif payment_method=="Electronic check":
    input_data["PaymentMethod_Electronic check"]=1
elif payment_method=="Mailed check":
    input_data["PaymentMethod_Mailed check"]=1
if multiple_lines=="Yes":
    input_data["MultipleLines_Yes"]=1
elif multiple_lines=="No phone service":
    input_data["MultipleLines_No phone service"]=1
if internet_service=="No":
    input_data["InternetService_No"]=1
    input_data["OnlineSecurity_No internet service"]=1
    input_data["OnlineBackup_No internet service"]=1
    input_data["DeviceProtection_No internet service"]=1
    input_data["TechSupport_No internet service"]=1
    input_data["StreamingTV_No internet service"]=1
    input_data["StreamingMovies_No internet service"]=1
elif internet_service=="Fiber optic":
    input_data["InternetService_Fiber optic"]=1
else:
    if online_security=="Yes":
       input_data["OnlineSecurity_Yes"]=1
    if online_backup=="Yes":
        input_data["OnlineBackup_Yes"]=1
    if device_protection=="Yes":
        input_data["DeviceProtection_Yes"]=1
    if tech_support=="Yes":
        input_data["TechSupport_Yes"]=1
    if streaming_tv=="Yes":
        input_data["StreamingTV_Yes"]=1
    if streaming_movies=="Yes":
        input_data["StreamingMovies_Yes"]=1

input_df=pd.DataFrame([input_data])

predict_button=st.button("Predict Churn")
if predict_button:
    input_scaled=scaler.transform(input_df)
    prediction=model.predict(input_scaled)
    probability=model.predict_proba(input_scaled)
    churn_probability=probability[0][1]
    shap_values=explainer(input_scaled)
    if prediction[0]==1:
        st.error("Customer is likely to churn")
    else:
        st.success("Customer is unlikely to churn")
    st.write(f"churn_probability: {churn_probability:.2%}")
    shap_row=shap_values.values[0]
    shap_df=pd.DataFrame({
        "Feature":feature_names,
        "SHAP_value":shap_row
    })
    shap_df["Abs_SHAP"]=(
        shap_df["SHAP_value"].abs()
    )
    shap_df=shap_df.sort_values(
        by="Abs_SHAP",
        ascending=False
    )
    top5= shap_df.head(5)
    st.subheader(
        "Why did the model make this prediction?"
    )
    for _, row in top5.iterrows():
        if row["SHAP_value"]>0:
            direction="Increased Churn Risk"
        else:
            direction="Reduced Churn Risk"
        st.write(
            f"{row['Feature']}:"
            f"{direction}"
            f"({row['SHAP_value']:.3f})"
        )



