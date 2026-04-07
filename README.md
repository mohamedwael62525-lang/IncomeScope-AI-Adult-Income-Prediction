# IncomeScope AI – Adult Income Prediction Dashboard
## Features

### Data Analysis & Exploration
- Dataset preview
- Interactive filtering
- Statistical summary
- Correlation heatmap
- Numerical distributions
- Categorical analysis
- Income class distribution visualization

### Feature Engineering
- Age grouping
- Capital gain/loss transformation
- Working hours grouping
- Education grouping
- Race grouping
- Country simplification

### Machine Learning
- Random Forest with GridSearchCV
- XGBoost with GridSearchCV
- SMOTE for handling class imbalance
- ROC Curve and Precision-Recall Curve
- Feature importance visualization
- Performance metrics:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - ROC-AUC

### Prediction System
Users can enter their own values and receive:
- Predicted income class
- Confidence score

### Built-in Chatbot
The dashboard includes a lightweight assistant capable of answering questions about:
- Number of rows and columns
- Missing values
- Model accuracy
- Best-performing model

---

## Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Matplotlib
- Plotly
- Scikit-learn
- Imbalanced-learn
- XGBoost

---

## Dataset

Dataset used: Adult Income Dataset from the UCI Machine Learning Repository.

The goal is to classify whether a person's income exceeds 50K per year.

---

2. Run the Streamlit application

streamlit run app.py


## How to Run the Project

1. Clone the repository

```bash
git clone https://github.com/your-username/IncomeScope-AI-Adult-Income-Prediction.git
cd IncomeScope-AI-Adult-Income-Prediction


