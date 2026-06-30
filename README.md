# KNN Classification & Noise/Outlier Analysis

Project for the **Machine Learning I** course (2024/2025), part of the 3rd year, 2nd semester of the Computer Science Bachelor's degree.

The goal of this project is to implement and evaluate a **K-Nearest Neighbors (KNN)** classifier, with a particular focus on studying how **noise and outliers** in the dataset affect its performance.

## Description

This project covers the implementation of the KNN algorithm, its application to classification tasks, and an evaluation of the model's robustness when the input data contains noisy points or outliers. Several sample datasets are included to test and validate the algorithm under different data conditions.

## 📁 Repository structure

| File / Folder | Description |
|---|---|
| `PL5_G7_KNN.ipynb` | Main notebook with the implementation of the KNN algorithm. |
| `PL5_G7_Avaliação_KNN.ipynb` | Notebook with the evaluation of the KNN model's performance. |
| `noise_outliers/` | Folder containing example datasets used to test the code under noisy/outlier conditions. |
| `PL5_G7_powerpoint.pdf` | Presentation slides explaining the work developed in this project. |
| `PracticalAssignment_ML1.pdf` | Original assignment outlining the tasks and requirements for the project. |

## How to run

1. Clone the repository:
   ```bash
   git clone https://github.com/anaamorim0/MachineLearning_Project.git
   cd MachineLearning_Project
   ```

2. Install the required dependencies (using a virtual environment is recommended):
   ```bash
   pip install numpy pandas scikit-learn matplotlib jupyter
   ```

3. Open the notebooks with Jupyter:
   ```bash
   jupyter notebook
   ```

4. Run `PL5_G7_KNN.ipynb` first to go through the KNN implementation, then `PL5_G7_Avaliação_KNN.ipynb` to see the model evaluation. The datasets in `noise_outliers/` can be used to test how the algorithm behaves with noisy data.

## Results

A summary of the methodology and results can be found in the presentation slides: [`PL5_G7_powerpoint.pdf`](./PL5_G7_powerpoint.pdf).

## Technologies

- Python
- Jupyter Notebook
- NumPy / Pandas
- Scikit-learn
- Matplotlib
