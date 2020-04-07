# %%
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('data'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# %%
train_data = pd.read_csv('data/train.csv')
train_data.head()

# %%
test_data = pd.read_csv('data/test.csv')
test_data.head()

# %%
women = train_data.loc[train_data.Sex == 'female']["Survived"]
rate_women = sum(women)/len(women)
print("% of women who survived:", rate_women)

# %%
men = train_data.loc[train_data.Sex == 'male']["Survived"]
rate_men = sum(men)/len(men)
print("% of men who survived", rate_men)

# %%
from sklearn.ensemble import RandomForestClassifier
y = train_data["Survived"]

features = ["Pclass", "Sex", "SibSp", "Parch"]
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X, y)
predictions = model.predict(X_test)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survivedd': predictions})
output.to_csv('my_submission.csv', index=False)
print("Your submission was successfully saved!")

# %%
from sklearn.metrics import precision_score, recall_score
train_predictions = model.predict(X)
precision_score(y, train_predictions)

# %%
recall_score(y, train_predictions)

# %%
from sklearn.metrics import f1_score
f1_score(y, train_predictions)

# %%
from sklearn.model_selection import cross_val_predict
y_train_pred = cross_val_predict(model, X, y, cv=3)
f1_score(y, y_train_pred)

# %%
train_data.info()

# %%
train_data['Sex'].value_counts()

# %%
train_data['Ticket'].value_counts()

# %%
train_data['Cabin'].value_counts()

# %%
train_data['Embarked'].value_counts()

# %%
train_data.describe()

# %%
%matplotlib inline
import matplotlib.pyplot as plt
train_data.hist(bins=50, figsize=(20,15))
plt.show()

# %%
features = ["Pclass", "Sex", "SibSp", "Parch", "Fare", "Age", "Embarked", "Survived"]
X = pd.get_dummies(train_data[features])
corr_matrix = X.corr()
corr_matrix["Survived"].sort_values(ascending=False)

# %%
from sklearn.base import BaseEstimator, TransformerMixin

class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names]

# %%
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([
    ('data_frame_selector', DataFrameSelector(['Age', 'RelativesOnboard', 'Fare'])),
    ('imputer', SimpleImputer(strategy="median")),
    # ('std_scaler', StandardScaler())
])

num_pipeline.fit_transform(train_data)

# %%
class MostFrequentImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.most_frequent_ = pd.Series([X[c].value_counts().index[0] for c in X], index=X.columns)
        return self
    def transform(self, X, y=None):
        return X.fillna(self.most_frequent_)

from sklearn.preprocessing import OneHotEncoder

cat_pipeline = Pipeline([
    ('select_cat', DataFrameSelector(['Pclass', 'Sex', 'Embarked', 'AgeBucket'])),
    ('imputer', MostFrequentImputer()),
    ('cat_encoder', OneHotEncoder(sparse=False))
])

cat_pipeline.fit_transform(train_data)

# %%
from sklearn.pipeline import FeatureUnion

preprocess_pipeline = FeatureUnion(transformer_list=[
    ('num_pipeline', num_pipeline),
    ('cat_pipeline', cat_pipeline)
])

X_train = preprocess_pipeline.fit_transform(train_data)
X_train

# %%
y_train = train_data['Survived']
y_train

# %%
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)

y_train_pred = cross_val_predict(model, X_train, y_train)
f1_score(y, y_train_pred)

# %%
train_data["AgeBucket"] = train_data["Age"] // 15 * 15
test_data["AgeBucket"] = test_data["Age"] // 15 * 15
train_data[["AgeBucket", "Survived"]].groupby(['AgeBucket']).mean()

# %%
train_data["RelativesOnboard"] = train_data["SibSp"] + train_data["Parch"]
test_data["RelativesOnboard"] = test_data["SibSp"] + test_data["Parch"]
train_data[["RelativesOnboard", "Survived"]].groupby(['RelativesOnboard']).mean()

# %%
train_data["Alone"] = train_data["SibSp"] + train_data["Parch"] == 0

corr_matrix = train_data.corr()
corr_matrix["Survived"].sort_values(ascending=False)


# %%
from sklearn.model_selection import GridSearchCV

param_grid = [
    {'n_estimators': [20, 30, 40, 50, 60, 70], 'max_features': [5, 6], 'max_depth': [10, 15, 20], 'bootstrap': [True, False]}
]

forest_clf = RandomForestClassifier()

grid_search = GridSearchCV(forest_clf, param_grid, cv=5, scoring='f1', verbose=3)

grid_search.fit(X_train, y_train)

grid_search.best_score_

# %%
grid_search.best_params_

# %%
X_test = preprocess_pipeline.fit_transform(test_data)
predictions = grid_search.best_estimator_.predict(X_test)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('my_submission.csv', index=False)
print("Your submission was successfully saved!")

# %%
