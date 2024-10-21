---

# Lung Cancer Prediction Tool

## Overview

The Lung Cancer Prediction Tool is a web-based application that utilizes advanced machine learning models to predict the probability of lung cancer based on user input. This tool aims to provide an intuitive and user-friendly interface for individuals seeking to assess their lung cancer risk.

## Features

- **Model Selection:** Users can choose between multiple machine learning models, including:
  - Logistic Regression
  - Random Forest
  - Decision Tree
  - SVC/SVR
- **Problem Type:** Users can select between classification and regression models based on their analysis needs.
- **Interactive Interface:** A simple and user-friendly web interface for easy interaction with the tool and interpretation of results.

## Tech Stack

- **Frontend:** 
  - HTML
  - CSS (Tailwind CSS)
  - JavaScript
- **Backend:**
  - Python (Flask)
- **Machine Learning Models:**
  - Scikit-learn
- **Deployment:** 
  - Render

## Installation

To run the project locally, follow these steps:

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/lung-cancer-prediction-tool.git
   cd lung-cancer-prediction-tool
   ```

2. **Set up a virtual environment:**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install the required packages:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application:**

   ```bash
   python app.py
   ```

5. **Open your web browser and go to:**

   ```
   http://127.0.0.1:5000/
   ```

## Usage

1. Select your gender and input your age.
2. Provide answers to various health-related questions.
3. Choose the machine learning model and the type of analysis (classification or regression).
4. Click on the submit button to see the predicted lung cancer probability.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any suggestions or improvements.


## Acknowledgments

- [Scikit-learn](https://scikit-learn.org/stable/) for the machine learning models.
- [Flask](https://flask.palletsprojects.com/) for the web framework.
- [Render](https://render.com/) for deployment.

