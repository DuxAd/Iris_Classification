####### Data Visualisation #########
import pandas as pd
import matplotlib.pyplot as plt
import MyFunction

df = pd.read_csv('Iris.csv')

print(df.keys())
# ['Id', 'SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm','Species']

# mappage = {'Iris-setosa': 10, 'Iris-versicolor': 20, 'Iris-virginica': 30}
# df['Species_int'] = df['Species'].map(mappage)

# Species = df['Species'].unique()
# Color = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'w']
# Dic = {Species: Color for Species, Color in zip(Species, Color[:len(Species)])}

# df['Species_int'] = df['Species'].map(Dic)

# plt.figure(figsize=(12, 8))

# plt.subplot(4, 4, 1)
# for i in range(len(Species)):
#     plt.hist(df[df['Species'] == Species[i]]['SepalLengthCm'], bins=10,color=Color[i], alpha=0.7)
# plt.xlabel('SepalLengthCm')
# plt.subplot(4, 4, 2)
# plt.scatter(df['SepalWidthCm'], df['SepalLengthCm'], c=df['Species_int'])
# plt.xticks([])
# plt.ylabel('SepalLengthCm')
# plt.subplot(4, 4, 3)
# plt.scatter(df['PetalLengthCm'], df['SepalLengthCm'], c=df['Species_int'])
# plt.xticks([])
# plt.yticks([])
# plt.subplot(4, 4, 4)
# plt.scatter(df['PetalWidthCm'], df['SepalLengthCm'], c=df['Species_int'])
# plt.xticks([])
# plt.yticks([])


# plt.subplot(4, 4, 6)
# for i in range(len(Species)):
#     plt.hist(df[df['Species'] == Species[i]]['SepalWidthCm'], bins=10,color=Color[i], alpha=0.7)
# plt.xlabel('SepalWidthCm')
# plt.subplot(4, 4, 7)
# plt.scatter(df['PetalLengthCm'], df['SepalWidthCm'], c=df['Species_int'])
# plt.xticks([])

# plt.ylabel('SepalWidthCm')
# plt.subplot(4, 4, 8)
# plt.scatter(df['PetalWidthCm'], df['SepalWidthCm'], c=df['Species_int'])
# plt.xticks([])
# plt.yticks([])
         
# plt.subplot(4, 4, 11)
# for i in range(len(Species)):
#     plt.hist(df[df['Species'] == Species[i]]['PetalLengthCm'], bins=10,color=Color[i], alpha=0.7)
# plt.xlabel('PetalLengthCm')
# plt.subplot(4, 4, 12)
# plt.scatter(df['PetalWidthCm'], df['PetalLengthCm'], c=df['Species_int'])
# plt.ylabel('PetalLengthCm')
# plt.xticks([])

# plt.subplot(4, 4, 16)
# for i in range(len(Species)):
#     plt.hist(df[df['Species'] == Species[i]]['PetalWidthCm'], bins=5,color=Color[i], alpha=0.7)
# plt.xlabel('PetalWidthCm')
import seaborn as sns
sns.pairplot(df, vars=['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm'], hue='Species')
plt.show()

####### Model #########
from sklearn.model_selection import train_test_split

features = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
X = df[features]
y = df['Species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,  random_state=42)

# from sklearn.preprocessing import MinMaxScaler

# scaler = MinMaxScaler()
# scaler.fit(X_train)

# X_train_np = scaler.transform(X_train)
# X_test_np = scaler.transform(X_test)

# X_train = pd.DataFrame(X_train_np, columns = features)
# X_test = pd.DataFrame(X_test_np, columns = features)
####### Decision Tree #########
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold

# Searching the best parameters 
param_grid = {
    'min_samples_split': [2, 5, 10],
    'max_depth': [None, 5, 10],
    'max_leaf_nodes': [22,25,27],
    'class_weight': [None, 'balanced']
}
grid_search = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42), scoring='accuracy')
grid_search.fit(X_train, y_train)
print("Meilleurs paramètres :", grid_search.best_params_)

# Creating and training the model
clf = DecisionTreeClassifier(max_depth=None, max_leaf_nodes= 22,
                             min_samples_split= 5, random_state=42)
clf.fit(X_train, y_train)

MyFunction.AffichageRes(clf, X_train, X_test, y_train, y_test)

feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': clf.feature_importances_,
}).sort_values('importance', ascending=False)
print("\n####################################")
print("Feature importance for the DecisionTree")
print(feature_importance)
print("\n####################################")


# Tree Visualisation
from sklearn.tree import DecisionTreeClassifier, export_text
text_representation = export_text(clf, feature_names=list(X.columns))
print(text_representation)

####### Random Tree #########
from sklearn.ensemble import RandomForestClassifier

# Searching the best parameters 
param_grid = {
    'min_samples_split': [2,3],
    'max_depth': [2,5],
    'n_estimators': [5, 10, 15],
    'min_samples_leaf': [2,3],
   # 'class_weight': [None, 'balanced', {0:1, 1:1.2}, {0:1, 1:1.5}]
}
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42), scoring='accuracy')
grid_search.fit(X_train, y_train)
print("Meilleurs paramètres :", grid_search.best_params_)

rfc = RandomForestClassifier(min_samples_leaf= 2, n_estimators=5, max_depth=2,
                             min_samples_split= 2,random_state=42)

rfc.fit(X_train, y_train)

MyFunction.AffichageRes(rfc, X_train, X_test, y_train, y_test)

feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': rfc.feature_importances_,
}).sort_values('importance', ascending=False)
print("\n####################################")
print("Feature importance for the DecisionTree")
print(feature_importance)
print("\n####################################")

##### Support Vector Machine #####
from sklearn import svm

param_grid_SVC = {
    'kernel': ['linear', 'poly', 'rbf'],
    'C': [1,5,10],
    'gamma': ['scale', 0.1, 0.01],
    'class_weight': [None, 'balanced']
}
grid_search = GridSearchCV(svm.SVC(random_state=42), param_grid_SVC, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42), scoring='accuracy')
grid_search.fit(X_train, y_train)
print("Meilleurs paramètres :", grid_search.best_params_)

SVC = svm.SVC(probability=True, kernel='rbf', C=10, random_state=42,
              class_weight=None, gamma='scale')
SVC.fit(X_train, y_train)

MyFunction.AffichageRes(SVC, X_train, X_test, y_train, y_test)

from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt

perm_imp = permutation_importance(SVC, X_test, y_test, n_repeats=10, random_state=42)
print("\n####################################")
print("Features importance for SVM")
for i in perm_imp.importances_mean.argsort()[::-1]:
    print(f"{X_test.columns[i]} \t:\t"
          f"{perm_imp.importances_mean[i]:.3f}"
          f" +/- {perm_imp.importances_std[i]:.3f}")
print("\n####################################")

##### KNN #####
from sklearn.neighbors import KNeighborsClassifier

param_grid_knn = {'n_neighbors': [1, 3, 5, 7],
                  'weights': ['uniform', 'distance']}
grid_search_knn = GridSearchCV(KNeighborsClassifier(), param_grid_knn, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42), scoring='accuracy')
grid_search_knn.fit(X_train, y_train)
print("Meilleurs paramètres KNN :", grid_search_knn.best_params_)

knn = KNeighborsClassifier(n_neighbors =5, weights='uniform')
knn.fit(X_train, y_train)

MyFunction.AffichageRes(knn, X_train, X_test, y_train, y_test)

perm_imp = permutation_importance(knn, X_test, y_test, n_repeats=10, random_state=42)
print("\n####################################")
print("Features importance for KNN")
for i in perm_imp.importances_mean.argsort()[::-1]:
    print(f"{X_test.columns[i]} \t:\t"
          f"{perm_imp.importances_mean[i]:.3f}"
          f" +/- {perm_imp.importances_std[i]:.3f}")
print("\n####################################")
