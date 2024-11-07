import pandas as pd

import torch
import torch.nn as nn

from sklearn.compose import make_column_transformer
from sklearn.preprocessing import LabelEncoder, StandardScaler, \
    OrdinalEncoder, Normalizer

from opacus import PrivacyEngine
from torch.utils.data import DataLoader, Dataset


class AdultDataSet(Dataset):

    def __init__(self, adult_data_file):
        header = ['age', 'workclass', 'fnlwgt', 'education', 'education_num',
                  'marital_status', 'occupation', 'relationship',
                  'race', 'sex', 'capital_gain', 'capital_loss',
                  'hours_per_week', 'native_country', 'income']
        df = pd.read_csv(adult_data_file, header=None, names=header,
                         sep=',\\s', na_values=['?'], engine='python')
        df = df.dropna()
        df = df.reset_index(drop=True)
        df['income'] = df['income'].apply(lambda x: x.replace('.', ''))

        categorical_columns = ['workclass', 'education', 'marital_status',
                               'occupation', 'relationship', 'race', 'sex',
                               'native_country']
        numerical_columns = ['age', 'capital_gain',
                             'capital_loss', 'hours_per_week']

        column_transformer = make_column_transformer(
            (OrdinalEncoder(), categorical_columns),
            (StandardScaler(), numerical_columns),
        )

        self.y = LabelEncoder().fit_transform(df['income']).astype(float)
        self.X = column_transformer.fit_transform(df)
        self.X = Normalizer().fit_transform(self.X)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class AdultClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(AdultClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size, dtype=torch.float64)
        self.fc2 = nn.Linear(hidden_size, output_size, dtype=torch.float64)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x


training_data = AdultDataSet('adult.data')
test_data = AdultDataSet('adult.test')
training_data_loader = DataLoader(training_data, batch_size=10, shuffle=True)
testing_data_loader = DataLoader(test_data, batch_size=1000)

input_size = len(next(iter(training_data[0])))
hidden_size = 250
output_size = 1

model = AdultClassifier(input_size, hidden_size, output_size)

criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

privacy_engine = PrivacyEngine()

model, optimizer, dataloader = privacy_engine.make_private(
    module=model,
    optimizer=optimizer,
    data_loader=training_data_loader,
    noise_multiplier=1.0,
    max_grad_norm=1.0,
)


epochs = 10
for epoch in range(epochs):
    for i, data in enumerate(training_data_loader):
        X, y = data

        optimizer.zero_grad()

        output = model(X)

        loss = criterion(output, y.reshape(y.size(dim=0), 1))
        loss.backward()

        optimizer.step()

    epsilon = privacy_engine.get_epsilon(delta=1e-3)

    print(f'Epoch [{epoch + 1}/{epochs}]')
    print(f'Loss: {loss.item():.4f}, Epsilon: {epsilon:.4f}')


with torch.no_grad():

    accuracy = 0.0
    batch_count = 0

    for i, test_data in enumerate(testing_data_loader):
        X_test, y_test = test_data
        test_output = model(X_test)
        test_output = torch.where(test_output > 0.5, 1, 0).resize(
            test_output.size(dim=0),)

        a_num = torch.sum(torch.where(test_output == y_test, 1, 0)).item()
        a_denom = y_test.size(dim=0)

        batch_accuracy = a_num / a_denom
        accuracy += batch_accuracy
        batch_count = i

    accuracy = accuracy / (batch_count+1)
    print(f'\nAccuracy: {accuracy * 100:.2f}%')
