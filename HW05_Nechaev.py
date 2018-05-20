
# coding: utf-8

# In[1]:


from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, precision_recall_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer

param_grid = {
    'n_estimators': [10, 20, 30, 40, 50],
    'max_depth': [None, 5, 10, 15, 20],
    'criterion': ['entropy', 'gini']
}

X_data, y_data = load_breast_cancer(return_X_y=True)

estimator = RandomForestClassifier(random_state=42)

print('Accuracy best params and score')
result = GridSearchCV(estimator, param_grid, cv=3, scoring='accuracy').fit(X_data, y_data)
print('\tParams:', result.best_params_)
print('\tScore:', result.best_score_)

def my_scorer(y_true, y_pred):
    precision, recall, _ = precision_recall_curve(y_true, y_pred[:, 1])
    return max([p for p, r in zip(precision, recall) if p < 1.5 * r and r > 0.5])

scorer = make_scorer(my_scorer, greater_is_better=True, needs_proba=True)

print('Custom loss best params and score')
result = GridSearchCV(estimator, param_grid, cv=3, scoring=scorer).fit(X_data, y_data)
print('\tParams:', result.best_params_)
print('\tScore:', result.best_score_)


# In[2]:


print(round(result.best_score_, 4))

