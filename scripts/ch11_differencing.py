import pandas as pd
df = pd.read_csv("teacher_survey.csv")
df.columns = ['name',
              'sex',
              'age',
              'maritalStatus',
              'hasChildren',
              'highestEducationLevel',
              'sourceOfStress',
              'smoker',
              'optimism',
              'lifeSatisfaction',
              'selfEsteem']

# Naively "anonymize" by removing the name column
del df["name"]

df.loc[(df['sex'] == 3) & (df['age'] == 27)]

df.loc[(df['sex'] == 3) & (df['age'] == 27)]['smoking'].sum()

predicate_a = (df['maritalStatus'] == 'Un-Married')
predicate_b = ((df['maritalStatus'] == 'Un-Married') | \
    (df['sex'] == 3) & (df['age'] == 27))
df.loc[predicate_a]["smoker"].sum() - df.loc[predicate_b]["smoker"].sum()
