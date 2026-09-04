Project Overview

A Streamlit web application that predicts student performance using a trained K-Nearest Neighbors (KNN) Machine Learning model (model.pkl). The app features an interactive dashboard to input subject marks, dynamic score metrics, and visual feedback upon prediction.

Key Features

Interactive Input Dashboard: Clean multi-column layout for entering scores across Hindi, English, Science, Maths, History, and Geography.

Automatic Aggregation: Dynamically calculates total and average scores.

Instant ML Predictions: Utilizes a pre-trained KNeighborsClassifier to predict final performance outcomes based on user inputs.

Visual Feedback: Custom CSS styling, loading animations, metric summary widgets, and balloon effects on prediction.

Tech Stack

Frontend/UI: Streamlit

Data Processing: NumPy

Machine Learning: Scikit-Learn (K-Nearest Neighbors)

Quick Start

Clone the repository and navigate to the project directory:

Bash
git clone <repository-url>
cd <repository-folder>
Install dependencies:

Bash
pip install -r requirements.txt
Ensure model.pkl is in the root directory.

Run the application:

Bash
streamlit run app.py
