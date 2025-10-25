
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

This project uses the Cleveland Heart Disease Dataset, which is part of the UCI Machine Learning Repository's Heart Disease Database.
Among the four available sources (Cleveland, Hungary, Switzerland, and Long Beach), the Cleveland dataset is the most commonly used in research because it contains the most complete and reliable records.

## Results

The performance of multiple Machine Learning algorithms was evaluated on the Cleveland Heart Disease dataset.  
Each model was assessed using key classification metrics: **Accuracy**, **Precision**, **Recall**, **F1-Score**, and **AUC**.

### Base Models

| Model | Accuracy | Precision | Recall | F1-Score | AUC |
|-------|----------|-----------|---------|-----------|-----|
| Logistic Regression | 0.8852 | 0.8387 | 0.9286 | 0.8814 | 0.9524 |
| Random Forest | **0.9016** | **0.8438** | **0.9643** | **0.9000** | **0.9594** |
| SVM | 0.8525 | 0.7879 | 0.9286 | 0.8525 | 0.9437 |
| XGBoost | 0.8689 | 0.8125 | 0.9286 | 0.8667 | 0.9481 |
| Neural Network | 0.8689 | 0.8125 | 0.9286 | 0.8667 | 0.9513 |

### Ensemble Models

| Model | Accuracy | Precision | Recall | F1-Score | AUC |
|-------|----------|-----------|---------|-----------|-----|
| Voting | 0.8525 | 0.7714 | **0.9643** | 0.8571 | **0.9502** |
| Stacking | 0.8525 | 0.8065 | 0.8929 | 0.8475 | 0.9177 |
| Full Stacking | **0.8852** | **0.8621** | 0.8929 | **0.8772** | 0.9437 |

### Key Findings

1. **Base Models**:
   - Random Forest menunjukkan performa terbaik di semua metrik
   - Logistic Regression dan Neural Network menunjukkan performa yang konsisten
   - SVM memiliki recall tinggi namun precision lebih rendah

2. **Ensemble Models**:
   - Full Stacking mencapai akurasi dan precision tertinggi
   - Voting Classifier unggul dalam recall
   - Semua model ensemble mencapai AUC > 0.91

3. **Overall Best**:
   - Random Forest (Base Model): Akurasi 90.16%, F1-Score 0.9000, AUC 0.9594
   - Full Stacking (Ensemble): Akurasi 88.52%, F1-Score 0.8772, AUC 0.9437
