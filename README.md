# Iris Classification

## Description
This project classifies Iris flowers into three species (Iris-setosa, Iris-versicolor, Iris-virginica) using the Iris dataset which can be found here : https://archive.ics.uci.edu/ml/datasets/iris. The goal is to demonstrate machine learning skills through data visualization, feature engineering, model training, and evaluation.

The project includes:
- **Data Visualization**: Histograms and scatter plots to explore feature distributions and class separability using seaborn.
- **Modeling**: Training of four models (DecisionTree, RandomForest, SVM and KNN) with hyperparameter tuning via `GridSearchCV`.
- **Feature Importance**: Analysis of feature importance.
- **Evaluation**: Metrics (accuracy, precision, recall, F1-score, ROC-AUC), cross-validation, and normalized confusion matrices.

## Results

| Model                | Accuracy (Test) | Accuracy (CV) | F1-Score | ROC-AUC | Precision | Recall | CV Std Dev |
|----------------------|-----------------|---------------|----------|---------|-----------|--------|------------|
| DecisionTree         | 1.000           | 0.942 ± 0.043 | 1.000    | 1.000   | 1.000     | 1.000  | 0.043      |
| RandomForest         | 1.000           | 0.942 ± 0.057 | 1.000    | 1.000   | 1.000     | 1.000  | 0.057      |
| SVM                  | 1.000           | 0.967 ± 0.017 | 1.000    | 1.000   | 1.000     | 1.000  | 0.017      |
| KNN                  | 1.000           | 0.958 ± 0.026 | 1.000    | 1.000   | 1.000     | 1.000  | 0.026      |

#### ** WIP ** ########
### Observations
- **DecisionTree** (max_depth=None, max_leaf_nodes=22, min_samples_split=5): Perfect test accuracy (1.000), but lower CV accuracy (0.942) with moderate variance (0.043), indicating slight overfitting due to small dataset size.
- **RandomForest** (max_depth=2, min_samples_leaf=2, min_samples_split=2, n_estimators=5): Perfect test accuracy (1.000), but CV accuracy (0.942) with higher variance (0.057), likely due to low n_estimators.
- **SVM** (kernel='rbf', C=10, class_weight=None): Best CV accuracy (0.967) and lowest variance (0.017), confirming robust performance. Note: kernel='rbf' requires feature normalization for optimal results.
- **KNN** (n_neighbors=5, weights='uniform'): High test accuracy (1.000) and CV accuracy (0.958) with low variance (0.026), benefiting from the dataset's simplicity.


- **Feature Importance **: Feature importance confirms that `PetalLengthCm` and `PetalWidthCm` are the most discriminative features, as seen in scatter plots.
- **Visualization**: Scatter plots show clear separation of classes, especially with `PetalLengthCm` and `PetalWidthCm`, explaining the high performance across models.
