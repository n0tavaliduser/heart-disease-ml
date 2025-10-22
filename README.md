
# Application of Machine Learning Algorithms for Heart Disease Prediction Based on the Cleveland Dataset

Stevano Titondea Prayoga Putra

## Requirements

Python 3.8

## Install

```bash
  conda create -n <ENV_NAME> python=3.8
  conda activate <ENV_NAME>
  pip install -r requirements.txt
  python -m src.main
```

## Datasets

This project uses the Cleveland Heart Disease Dataset, which is part of the UCI Machine Learning Repository’s Heart Disease Database.
Among the four available sources (Cleveland, Hungary, Switzerland, and Long Beach), the Cleveland dataset is the most commonly used in research because it contains the most complete and reliable records.

## Results

The performance of four different Machine Learning algorithms was evaluated on the Cleveland Heart Disease dataset.  
Each model was assessed using key classification metrics: **Accuracy**, **Precision**, **Recall**, **F1-Score**, and **AUC**.

| Model | Accuracy | Precision | Recall | F1-Score | AUC |
|--------|-----------|------------|----------|-----------|--------|
| Logistic Regression | 0.8689 | 0.8125 | 0.9286 | 0.8667 | 0.9513 |
| **Random Forest** | **0.8852** | 0.8182 | **0.9643** | **0.8852** | **0.9513** |
| Support Vector Machine (SVM) | 0.8525 | 0.8065 | 0.8929 | 0.8475 | 0.9437 |
| XGBoost | **0.8852** | **0.8387** | 0.9286 | 0.8814 | 0.9351 |

## Run the Code

description

## Pretrained Model

description
