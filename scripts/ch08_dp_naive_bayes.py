import pandas as pd
from diffprivlib.models import GaussianNB


X_columns = ["age", "education_num", "capital_gain",
             "capital_loss ", "hours_per_week"]
Y_column = "income"

def load_data(file_name):
    df = pd.read_csv(file_name)
    return df[X_columns], df[Y_column].apply(lambda x: 1 if ">" in x else 0)

# this consumes ε=.01 on the individuals in the train data
dp_clf = GaussianNB(epsilon=0.01)
dp_clf.fit(*load_data("adult.data"))

# this consumes ε=∞ on the individuals in the test data (it is not privatized!)
mean_accuracy = dp_clf.score(*load_data("adult.test"))
