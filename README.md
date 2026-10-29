# CPI Data Analysis and Prediction

## Project Overview

This project analyzes Consumer Price Index (CPI) data using Python and applies machine learning regression algorithms to estimate Agricultural CPI based on Rural CPI.

The project combines data preprocessing, exploratory data analysis (EDA), data visualization, regression modelling, and performance evaluation.

## Objectives

- Analyze the relationship between Rural CPI and Agricultural CPI.
- Explore CPI distributions and year-wise trends.
- Compare CPI values across states.
- Build regression models to estimate Agricultural CPI.
- Evaluate models using MAE, MSE, and R² score.

## Technologies Used

- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- Git and GitHub

## Exploratory Data Analysis

The project includes:
- Rural CPI versus Agricultural CPI scatter plot
- Top five states by maximum Agricultural CPI
- Distribution of Agricultural CPI values
- Year-wise average CPI trend
- Three-year moving average
- Minimum versus maximum CPI by year
- Comparison of states with the highest and lowest mean CPI

## Machine Learning Models

Four regression algorithms were implemented:

1. Linear Regression
2. Decision Tree Regression
3. K-Nearest Neighbors (KNN) Regression
4. Support Vector Regression (SVR)

The predictor is `index_rl` (Rural CPI), and the target is `index_al` (Agricultural CPI).

## Model Evaluation

| Model | MAE | MSE | R² Score |
|---|---:|---:|---:|
| Linear Regression | 7.2751 | 152.3308 | 0.9985 |
| Decision Tree Regression | 11.6116 | 280.3189 | 0.9973 |
| KNN Regression | 7.5307 | 158.5076 | 0.9985 |
| Support Vector Regression | 7.1464 | 160.5191 | 0.9985 |

These results are from the project's recorded test-set evaluation. SVR has the lowest MAE, while Linear Regression has the lowest MSE. The displayed R² values are rounded.

## Dataset

The project uses CPI data identified as sourced from the Ministry of Statistics and Programme Implementation (MOSPI), Government of India.

Official website: https://www.mospi.gov.in/

The processed dataset shown in the project output contains 6,531 rows and 7 columns. The code applies a filter to retain observations where `index_al > 200`; therefore, the results describe the filtered dataset.

## How to Run

1. Clone or download this repository.
2. Install the required Python libraries.
3. Ensure the CPI CSV file is available at the path expected by the code.
4. Run the Python script.

Install dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

Run:

```bash
python cpi_analysis_prediction.py
```

Update the CSV path in the script if the dataset is stored elsewhere.

## Limitations

- Agricultural CPI is estimated using Rural CPI as the only predictor.
- Results depend on the preprocessing and filtering decisions.
- The model estimates are not automatically forecasts of future CPI.
- The reported evaluation scores should not be assumed to generalize to other datasets or time periods.

## Project Report

The repository includes the project report documenting the methodology, visualizations, regression models, evaluation metrics, and conclusions.# CPI-Data-Analysis-and-Prediction
