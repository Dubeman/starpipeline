from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold, cross_val_score 

X, y = datasets.load_iris(return_X_y=True)

clf = DecisionTreeClassifier(random_state=17)

k_folds = KFold(n_splits=5, shuffle=True, random_state=31)
scores = cross_val_score(clf, X, y, cv = k_folds)

print("Cross Validation Scores: ", scores)
print("Average CV Score: ", scores.mean())
print("Number of CV Scores used in Average: ", len(scores)) 