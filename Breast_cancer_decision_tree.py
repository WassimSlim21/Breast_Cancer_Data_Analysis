# LIBRARIES:
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.svm import SVC
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import classification_report
from sklearn.tree import plot_tree
from sklearn.model_selection import GridSearchCV
from sklearn.tree import export_text

# Breast cancer Data import 
df = load_breast_cancer()
df.keys()

#DF keys
print(df.keys())


# Breast Cancer Data SET diagnostic

print(df.DESCR)
print(df.target_names)

# Extract from `df`: 
# observations in : 2D array `x` 
# classes in a 1D array `y`.
x = df.data
y=df.target

print(x.shape)
print(y.shape)
#### Divide the set of observations X and the set of classes y, each into two subsets:
#- a learning subset: 70% of the initial set
#- a test subset: 30% of the initial set

# random_state=some_number to ensure that your split will always be the same
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=3)

## Create a dt instance of the DecisionTreeClassifier class, setting random_state to zero. Keep the Gini index as default criterion. ##

dt = DecisionTreeClassifier(random_state=0)
## Train the dt model on training subsets (observations and classes) ##

dt.fit(x_train, y_train)

#Calculate learning and test scores for model dt
print('The train score is :', dt.score(x_train, y_train))
print('The train score is :', dt.score(x_test, y_test))
# As can be seen from the training set, the correct classification rate is 100% --> an indicator of overlearning.
print(y_test.shape)

y_pred = dt.predict(x_test)

print("Condusion Matrix :", confusion_matrix(y_test, y_pred))
clf = SVC(random_state=0)
clf.fit(x_train, y_train)
predictions = clf.predict(x_test)

# Ensure that at least one instance of each class is predicted
unique_true_labels = unique_labels(y_test, predictions)
cm = confusion_matrix(y_test, predictions, labels=unique_true_labels, )

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=df.target_names)
disp.plot(cmap='Blues')
y_pred = dt.predict(x_test)
print("############### Classification ###############")
print(classification_report(y_test, y_pred, digits=8, target_names= df.target_names ))
print("X Shape : ", x.shape) 

print(" Feature Importance DT :", dt.feature_importances_) 
print(" Feature names DT:", df.feature_names) 
names = ['mean radius', 'mean texture', 'mean perimeter', 'mean area',
         'mean smoothness', 'mean compactness', 'mean concavity',
         'mean concave points', 'mean symmetry', 'mean fractal dimension',
         'radius error', 'texture error', 'perimeter error', 'area error',
         'smoothness error', 'compactness error', 'concavity error',
         'concave points error', 'symmetry error',
         'fractal dimension error', 'worst radius', 'worst texture',
         'worst perimeter', 'worst area', 'worst smoothness',
         'worst compactness', 'worst concavity', 'worst concave points',
         'worst symmetry', 'worst fractal dimension']
for var, importance in zip(names, dt.feature_importances_):
    if importance != 0:
        print(var, importance)
plt.figure(figsize=(30,10))
plot_tree(dt, feature_names= names, class_names=df.target_names, filled=True)
from sklearn.tree import export_text
print(export_text(dt, feature_names=names))
# The tree is too long and difficult to interpret. To deal with this problem, we'll try to set the hyperparameters of our model.
# Create a param_grid dictionary, in order to configure the following two hyperparameters:
# - criterion: which can be either the Gini index or entropy
# - max_depth: which varies from 1 to 9

param_grid = {'criterion': ['gini', 'entropy'],
             'max_depth': np.arange(1,10)}
# Create a grid search instance applied to the decision tree classification algorithm. The search must test all combinations of hyperparameter values
grid = GridSearchCV(DecisionTreeClassifier(random_state=1), param_grid=param_grid, cv=5)

# Train the created model instance on the appropriate data subsets.
grid.fit(x_train, y_train)
print("Grid Params :", grid.best_params_)
final_model = DecisionTreeClassifier(random_state=1, criterion='gini', max_depth=2)
final_model.fit(x_train, y_train)
print( 'le train_score=',final_model.score(x_train, y_train))
print( 'le test_score=',final_model.score(x_test, y_test))

plt.figure(figsize=(20,10))
plot_tree(final_model, feature_names= names, class_names=df.target_names, filled=True)

final_model.feature_importances_
print(export_text(final_model, feature_names=names))
plt.show()