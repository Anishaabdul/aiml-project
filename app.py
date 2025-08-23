import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import pickle

# --------------------------
# Step 1: Load Dataset
# --------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("creditcard.csv")
    return df

df = load_data()

# --------------------------
# Step 2: Train Model
# --------------------------
@st.cache_resource
def train_model():
    X = df.drop("Class", axis=1)
    y = df["Class"]

    scaler = StandardScaler()
    X["Amount"] = scaler.fit_transform(X[["Amount"]])

    model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
    model.fit(X, y)
    return model, scaler

model, scaler = train_model()

# --------------------------
# Step 3: Streamlit UI
# --------------------------
st.title("💳 Credit Card Fraud Detection App")

st.write("Enter transaction details to check if it might be **fraudulent**.")

# User Input
time = st.number_input("Transaction Time", min_value=0, value=1000)
amount = st.number_input("Transaction Amount ($)", min_value=0.0, value=50.0)

# PCA features (normally hidden, but we’ll let user enter randomly for demo)
pca_inputs = []
for i in range(1, 29):  # V1 to V28
    val = st.number_input(f"V{i}", value=0.0, step=0.1)
    pca_inputs.append(val)

# --------------------------
# Step 4: Make Prediction
# --------------------------
if st.button("Predict"):
    input_data = np.array([[time, amount] + pca_inputs])
    input_df = pd.DataFrame(input_data, columns=df.drop("Class", axis=1).columns)

    # Scale amount
    input_df["Amount"] = scaler.transform(input_df[["Amount"]])

    # Predict
    prediction = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.error(f"🚨 Fraud Detected! (Probability: {prob:.2f})")
    else:
        st.success(f"✅ Transaction is Safe. (Fraud Probability: {prob:.2f})")
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, roc_curve, auc
from xgboost import XGBClassifier

# --------------------------
# Step 1: Load Dataset
# --------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("creditcard.csv")
    return df

df = load_data()

# --------------------------
# Step 2: Train Model
# --------------------------
@st.cache_resource
def train_model():
    X = df[["Time", "Amount"]]
    y = df["Class"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
    model.fit(X_train, y_train)

    return model, scaler, X_test, y_test

model, scaler, X_test, y_test = train_model()

# --------------------------
# Step 3: Streamlit Tabs
# --------------------------
st.title("💳 Credit Card Fraud Detection Dashboard")

tab1, tab2, tab3 = st.tabs(["📊 Data Insights", "🤖 Prediction", "📈 Model Evaluation"])

# --------------------------
# 📊 Tab 1: Data Insights
# --------------------------
with tab1:
    st.subheader("Fraud vs Non-Fraud Transactions")
    fraud_count = df["Class"].value_counts()

    fig, ax = plt.subplots()
    sns.barplot(x=fraud_count.index, y=fraud_count.values, ax=ax, palette="viridis")
    ax.set_xticklabels(["Non-Fraud (0)", "Fraud (1)"])
    ax.set_ylabel("Count")
    st.pyplot(fig)

    st.write("Dataset Shape:", df.shape)
    st.write(df.describe())

# --------------------------
# 🤖 Tab 2: Prediction
# --------------------------
with tab2:
    st.subheader("Enter Transaction Details")

    time = st.number_input("Transaction Time", min_value=0, value=1000)
    amount = st.number_input("Transaction Amount ($)", min_value=0.0, value=50.0)

    if st.button("Predict Fraud"):
        input_data = np.array([[time, amount]])
        input_scaled = scaler.transform(input_data)

        prediction = model.predict(input_scaled)[0]
        prob = model.predict_proba(input_scaled)[0][1]

        if prediction == 1:
            st.error(f"🚨 Fraud Detected! (Probability: {prob:.2f})")
        else:
            st.success(f"✅ Transaction is Safe. (Fraud Probability: {prob:.2f})")

# --------------------------
# 📈 Tab 3: Model Evaluation
# --------------------------
with tab3:
    st.subheader("Confusion Matrix & ROC Curve")

    # Confusion Matrix
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Non-Fraud", "Fraud"], yticklabels=["Non-Fraud", "Fraud"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    # ROC Curve
    y_probs = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], "r--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    st.pyplot(fig)
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
