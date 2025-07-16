import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import streamlit as st 

# Load data
data = pd.read_csv("creditcard.csv")

legit = data[data.Class == 0]
fraud = data[data.Class == 1]

# Balance the dataset
legit_sample = legit.sample(n=len(fraud), random_state=2)
data = pd.concat([legit_sample, fraud], axis=0)

# Split data into features and target
X = data.drop(columns="Class", axis=1)
y = data["Class"]

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=2
)

# Train Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate model performance
train_acc = accuracy_score(model.predict(X_train), y_train)
test_acc = accuracy_score(model.predict(X_test), y_test)

st.title("Credit Card Fraud Detection Model")
# st.write(f"**Training Accuracy:** {train_acc:.2f} | **Testing Accuracy:** {test_acc:.2f}")

# Single input field
input_df = st.text_input("Enter All Required Features Values..")

submit = st.button("--SUBMIT--")

if submit:
    input_df_splited = input_df.split(',')
    cleaned_inputs = [item.strip() for item in input_df_splited if item.strip() != '']

    expected_features = X.shape[1]

    if len(cleaned_inputs) != expected_features:
        st.error(f"⚠️ Please enter exactly {expected_features} numeric values.")
    else:
        try:
            np_df = np.asarray(cleaned_inputs, dtype=np.float64)
            prediction = model.predict(np_df.reshape(1, -1))

            if prediction[0] == 0:
                st.success("Legitimate Transaction ✅")
            else:
                st.error("⚠️ Fraudulent Transaction Detected!")

        except ValueError:
            st.error("❌ Invalid input: All values must be numeric.")