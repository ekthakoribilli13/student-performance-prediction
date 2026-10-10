# Student Performance Prediction Using Machine Learning

## About the Project

This project focuses on predicting students final marks using machine learning. It considers three factors: attendance, daily study hours, and previous marks.

I built this project using Python and Scikit-learn, starting with data analysis and visualization and then developing a Linear Regression model. I also created a simple web application using Streamlit, where users can enter student details and view the predicted marks.

The project is currently based on a small synthetic dataset and is intended for learning and demonstrating the machine learning workflow.

## Project Objectives

- Analyze student data to understand the relationship between academic factors and final marks.
- Visualize the data using graphs.
- Build a Linear Regression model to predict final marks.
- Evaluate model performance using MAE and R² score.
- Develop an interactive interface for making predictions.

## Tools and Technologies

- **Python** – Main programming language
- **Pandas** – Data loading and analysis
- **Matplotlib** – Data visualization
- **Scikit-learn** – Model training and evaluation
- **Streamlit** – Web application
- **Joblib** – Saving and loading the trained model

## Features Used for Prediction

| Feature | Description |
|---|---|
| Attendance | Student's attendance percentage |
| Study Hours | Average hours spent studying per day |
| Previous Marks | Marks obtained previously |

**Target variable:** Final marks

## Project Structure


StudentPerformancePrediction/
├── app.py
├── main.py
├── train_model.py
├── student_data.csv
├── student_model.pkl
├── README.md
└── requirements.txt


## How to Run the Project

### 1. Clone or download the project

Open the project folder in VS Code or a terminal.

### 2. Create a virtual environment


python -m venv .venv


Activate it on Windows PowerShell:

powershell
.\.venv\Scripts\Activate.ps1


### 3. Install the required packages

python -m pip install -r requirements.txt


### 4. Train the model

python train_model.py


This trains the Linear Regression model, evaluates it on a test split, and saves it as `student_model.pkl`.

### 5. Launch the application

```bash
python -m streamlit run app.py
```

Open the local URL shown in the terminal to use the application.

## Model Evaluation

The model's performance is assessed using two metrics:

- **Mean Absolute Error (MAE):** Shows the average absolute difference between actual and predicted marks. A lower value is generally better.
- **R² Score:** Indicates how well the model explains variation in the target values. A higher value is generally better, although it must be interpreted alongside other metrics.

The current dataset contains only 20 synthetic records. Therefore, the evaluation results are preliminary and should not be treated as evidence of accuracy on real student data.

## Future Scope

- Test the model using a larger and more representative dataset.
- Apply cross-validation for a more reliable evaluation.
- Compare different regression algorithms.
- Improve the application's design and usability.
- Deploy the application online after further testing.

## Author

Developed as a machine learning project to gain practical experience in data analysis, regression, model evaluation, and application development.