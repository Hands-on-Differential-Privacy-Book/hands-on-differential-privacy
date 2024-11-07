import pandas as pd

from sklearn.preprocessing import LabelEncoder
from diffprivlib.models import LinearRegression


header = ['age', 'workclass', 'fnlwgt', 'education', 'education_num',
          'marital_status', 'occupation', 'relationship',
          'race', 'sex', 'capital_gain', 'capital_loss',
          'hours_per_week', 'native_country', 'income']
label = 'income'

df = pd.read_csv('adult.data', header=None, names=header,
                 sep=',\\s', na_values=['?'], engine='python')
df = df.dropna()
df = df.reset_index(drop=True)

categorical_columns = ['workclass', 'education', 'marital_status',
                       'occupation', 'relationship', 'race', 'sex',
                       'native_country']

df_train = pd.read_csv('adult.data')
df_test = pd.read_csv('adult.test')

for cat in categorical_columns:
    df_train[cat] = LabelEncoder().fit_transform(df_train[cat]).reshape(-1, 1)
    df_test[cat] = LabelEncoder().fit_transform(df_test[cat]).reshape(-1, 1)

y_train = df_train[label].apply(lambda x: 1 if '>' in x else 0)
y_test = df_test[label].apply(lambda x: 1 if '>' in x else 0)

predictors = ['age', 'workclass', 'education', 'marital_status',
              'occupation', 'relationship', 'race', 'sex',
              'hours_per_week', 'native_country', 'income']

X_train = df_train[predictors].drop(columns=['income'])
X_test = df_test[predictors].drop(columns=['income'])

# this consumes ε=1 on the individuals in the train data
regr = LinearRegression(epsilon=1.)
regr.fit(X_train, y_train)
releases = regr.coef_, regr.intercept_

# this consumes ε=∞ on the individuals in the test data (it is not privatized!)
r2_score = regr.score(X_test, y_test)

# For comparison, here is the same process with scikit-learn.
# Let's compare R2 values
from sklearn.linear_model import LinearRegression as ScikitLinearRegression

scikit_regr = ScikitLinearRegression()
scikit_regr.fit(X_train, y_train)
scikit_releases = regr.coef_, regr.intercept_

scikit_r2_score = scikit_regr.score(X_test, y_test)

print(f"R^2 with DP: {'': <5} {r2_score:.3f}")
print(f"R^2 without DP: {'': <2} {scikit_r2_score:.3f}")
