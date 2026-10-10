
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
# Load the dataset
data = pd.read_csv("student_data.csv")

# 1. Display the first five rows
print("FIRST FIVE ROWS")
print(data.head())

# 2. Check the number of rows and columns
print("\nDATASET SHAPE")
print(data.shape)

# 3. Display column names
print("\nCOLUMN NAMES")
print(data.columns)

# 4. Check data types and missing values
print("\nDATASET INFORMATION")
data.info()

# 5. Get basic statistical information
print("\nSTATISTICAL SUMMARY")
print(data.describe())

# 6. Count missing values in each column
print("\nMISSING VALUES")
print(data.isnull().sum())
# Graph 1: Study Hours vs Final Marks
plt.scatter(data["study_hours"], data["final_marks"])

plt.title("Study Hours vs Final Marks")
plt.xlabel("Study Hours per Day")
plt.ylabel("Final Marks")
plt.grid(True)
plt.show()


# Graph 2: Attendance vs Final Marks
plt.scatter(data["attendance"], data["final_marks"])

plt.title("Attendance vs Final Marks")
plt.xlabel("Attendance (%)")
plt.ylabel("Final Marks")
plt.grid(True)
plt.show()


# Graph 3: Previous Marks vs Final Marks
plt.scatter(data["previous_marks"], data["final_marks"])

plt.title("Previous Marks vs Final Marks")
plt.xlabel("Previous Marks")
plt.ylabel("Final Marks")
plt.grid(True)
plt.show()

# 7. Select features and target

X = data[["attendance", "study_hours", "previous_marks"]]
y = data["final_marks"]


# 8. Split data into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# 9. Create and train the model

model = LinearRegression()
model.fit(X_train, y_train)


# 10. Predict final marks

predictions = model.predict(X_test)


# 11. Evaluate the model

print("\nACTUAL MARKS:")
print(y_test.to_list())

print("\nPREDICTED MARKS:")
print(predictions.round(2))

print("\nMEAN ABSOLUTE ERROR:")
print(round(mean_absolute_error(y_test, predictions), 2))

print("\nR2 SCORE:")
print(round(r2_score(y_test, predictions), 2))

# Predict final marks for a new student

print("\n--- NEW STUDENT PREDICTION ---")

attendance = float(input("Enter attendance percentage: "))
study_hours = float(input("Enter daily study hours: "))
previous_marks = float(input("Enter previous marks: "))

new_student = pd.DataFrame({
    "attendance": [attendance],
    "study_hours": [study_hours],
    "previous_marks": [previous_marks]
})

predicted_marks = model.predict(new_student)

print("Predicted final marks:", round(predicted_marks[0], 2))


