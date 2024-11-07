import pandas as pd
from sklearn.compose import make_column_transformer

from snsynth import Synthesizer

from sklearn.preprocessing import OrdinalEncoder, StandardScaler, LabelEncoder, Normalizer

header = ['age', 'workclass', 'fnlwgt', 'education', 'education_num',
          'marital_status', 'occupation', 'relationship',
          'race', 'sex', 'capital_gain', 'capital_loss',
          'hours_per_week', 'native_country', 'income']

df = pd.read_csv('adult.data', header=None, names=header,
                 sep=',\\s', na_values=['?'], engine='python')
df = df.dropna()
df = df.reset_index(drop=True)
df['income'] = df['income'].apply(lambda x: x.replace('.', ''))

categorical_columns = ['workclass', 'income']
numerical_columns = ['age', 'hours_per_week']

column_transformer = make_column_transformer(
    (OrdinalEncoder(), categorical_columns),
    (StandardScaler(), numerical_columns),
)

X = column_transformer.fit_transform(df)
X = Normalizer().fit_transform(X)

synth = Synthesizer.create('mwem', epsilon=1.0)
sample = synth.fit_sample(
        df[['workclass',
            'age',
            'income',
            'hours_per_week']],
        preprocessor_eps=0.2)

sample_conditional = synth.sample_conditional(2, "age < 65 AND income > 0")

print(sample_conditional)
