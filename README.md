# AI In Business (Finance) & AutoML

## Project Overview
This project utilizes **XGBoost** and **AutoML** for predicting credit card defaults. The dataset used is sourced from **UCI Credit Card Dataset**, and the model is deployed using **Amazon SageMaker**.

## Features
- **Data Preprocessing**: One-hot encoding, normalization, and feature engineering.
- **Exploratory Data Analysis (EDA)**: Correlation matrix, box plots, and count plots.
- **Model Training**: XGBoost classifier with hyperparameter tuning.
- **AutoML & SageMaker**: Deploying the model on AWS cloud.

## Dataset
- `1.1_UCI_Credit_Card.csv` (not included in the repository)
- **Target Variable**: `default.payment.next.month` (1 = Default, 0 = Non-Default)

## Installation
Clone the repository:
```bash
git clone https://github.com/kerolous-samir/AI_In_Business_Finance_AutoML.git
cd AI_In_Business_Finance_AutoML
```

Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage
### Data Preprocessing
```python
from src.preprocess import preprocess_data
preprocess_data("data/1.1_UCI_Credit_Card.csv")
```

### Model Training
```python
from src.train import train_xgboost
train_xgboost()
```

### AutoML Deployment (AWS SageMaker)
```python
python src/deploy.py
```

## Results
- **Accuracy**: X%
- **Precision**: Y%
- **Recall**: Z%

## Repository Structure
```
AI_In_Business_Finance_AutoML/
│── README.md
│── data/
│   │── 1.1_UCI_Credit_Card.csv  (ignored in .gitignore)
│── notebooks/
│   │── analysis.ipynb
│── src/
│   │── preprocess.py
│   │── train.py
│   │── deploy.py
│── main.py
│── requirements.txt
│── LICENSE
│── .gitignore
```

## License
This project is licensed under the MIT License.

## Author
[Kerolous Samir](https://github.com/kerolous-samir)
