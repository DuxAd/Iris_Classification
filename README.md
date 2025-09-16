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
| DecisionTree         | 0.920           | 0.970 ± 0.054 | 0.917    | 0.939   | 0.917     | 0.917  | 0.046      |
| RandomForest         | 0.960           | 0.950 ± 0.050 | 0.965    | 0.990   | 0.965     | 0.968  | 0.067      |
| SVM                  | 0.980           | 0.990 ± 0.040 | 0.982    | 0.998   | 0.981     | 0.984  | 0.030      |
| KNN                  | 0.960           | 0.980 ± 0.046 | 0.965    | 0.985   | 0.965     | 0.968  | 0.060      |

#### ** WIP ** ########
### Observations
- **DecisionTree** (`max_depth=None`, `max_leaf_nodes=22`, `min_samples_split=2`): Lowest test accuracy (0.940) and CV accuracy (0.950).
- **RandomForest** (`max_depth=5`, `min_samples_leaf=3`, `min_samples_split=2`, `n_estimators=40`): Test accuracy of 0.960 and stable CV (0.950).
- **SVM** (`kernel='rbf'`, `C=10`, `class_weight='balanced'`): Best CV accuracy (0.990) and high ROC-AUC (0.998), confirming linear separability of classes. 
- **KNN** (`n_neighbors=3`): High test accuracy (0.960), excellent CV accuracy (0.980).
- **Feature Importance **: Permutation importance confirms that `PetalLengthCm` and `PetalWidthCm` are the most discriminative features, as seen in scatter plots.
- **Visualization**: Scatter plots show clear separation of classes, especially with `PetalLengthCm` and `PetalWidthCm`, explaining the high performance across models.
