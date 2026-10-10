
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# ---------------- PAGE SETTINGS ----------------
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide"
)

# ---------------- LOAD DATA ----------------
data = pd.read_csv("student_data.csv")

# ---------------- TRAIN MODEL ----------------
X = data[["attendance", "study_hours", "previous_marks"]]
y = data["final_marks"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = joblib.load("student_model.pkl")

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

# ---------------- HEADER ----------------
st.title("🎓 Student Performance Predictor")
st.write(
    "An interactive machine learning application "
    "for estimating student final marks."
)

st.divider()

# ---------------- SIDEBAR ----------------
st.sidebar.header("Student Details")
st.sidebar.write("Enter the student's academic information.")

attendance = st.sidebar.slider(
    "Attendance (%)", 0, 100, 90
)

study_hours = st.sidebar.slider(
    "Daily Study Hours", 0, 12, 4
)

previous_marks = st.sidebar.slider(
    "Previous Marks", 0, 100, 80
)

# ---------------- PREDICTION ----------------
st.subheader("📊 Prediction Dashboard")

if st.button("Predict Final Marks", type="primary"):
    new_student = pd.DataFrame({
        "attendance": [attendance],
        "study_hours": [study_hours],
        "previous_marks": [previous_marks]
    })

    result = model.predict(new_student)[0]
    result = max(0, min(100, result))

    st.metric("Estimated Final Marks", f"{result:.2f} / 100")

    if result >= 85:
        st.success("High estimated score")
    elif result >= 60:
        st.info("Moderate estimated score")
    else:
        st.warning("Lower estimated score")

    st.caption(
        "This estimate is based on a small synthetic dataset "
        "and is not a reliable real-world assessment."
    )

st.divider()

# ---------------- MODEL EVALUATION ----------------
st.subheader("📈 Model Evaluation")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Dataset Records", len(data))

with col2:
    st.metric("Mean Absolute Error", f"{mae:.2f} marks")

with col3:
    st.metric("R² Score", f"{r2:.2f}")

st.caption(
    "Evaluation uses a held-out test set. Results may vary "
    "with different data splits and datasets."
)

# ---------------- DATA VISUALIZATION ----------------
st.divider()
st.subheader("📉 Explore Student Data")

selected_feature = st.selectbox(
    "Choose a feature to compare with final marks",
    ["attendance", "study_hours", "previous_marks"]
)

fig, ax = plt.subplots()

ax.scatter(
    data[selected_feature],
    data["final_marks"]
)

ax.set_xlabel(selected_feature.replace("_", " ").title())
ax.set_ylabel("Final Marks")
ax.set_title(
    f"{selected_feature.replace('_', ' ').title()} vs Final Marks"
)
ax.grid(True, alpha=0.3)

st.pyplot(fig)
plt.close(fig)

# ---------------- DATA PREVIEW ----------------
with st.expander("View Dataset"):
    st.dataframe(data, use_container_width=True)

st.caption(
    "Student Performance Prediction | Educational project"
)

# ---------------- MODEL COMPARISON ----------------
st.divider()
st.subheader("🤖 Machine Learning Model Comparison")

models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )
}

comparison_results = []

for name, algorithm in models.items():
    algorithm.fit(X_train, y_train)
    model_predictions = algorithm.predict(X_test)

    model_mae = mean_absolute_error(
        y_test, model_predictions
    )
    model_r2 = r2_score(y_test, model_predictions)

    comparison_results.append({
        "Model": name,
        "MAE": round(model_mae, 2),
        "R² Score": round(model_r2, 2)
    })

comparison_df = pd.DataFrame(comparison_results)

st.dataframe(
    comparison_df,
    use_container_width=True,
    hide_index=True
)

st.caption(
    "Lower MAE is better. Higher R² is generally better. "
    "With only 20 synthetic records, this comparison is "
    "for learning and is not reliable evidence of real-world performance."
)
