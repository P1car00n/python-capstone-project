import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


class Model:

    def __init__(self, description, model):
        self.description = description
        self.model = model

    def get_prediction(self, samples):
        return self.model.predict(samples)

    def get_score(self, X, y):
        return self.model.score(X, y)

    def __repr__(self) -> str:
        return self.description


class LRM(Model):

    def __init__(self, X, y, description='linear regression model', **kwargs):
        Model.__init__(self, description,
                       model=LinearRegression(**kwargs).fit(X, y))


if __name__ == '__main__':
    # getting the data
    dataset_train = pd.read_parquet(
        input('Path to the train dataset: '))
    dataset_test = pd.read_parquet(
        input('Path to the test dataset: '))
    X_train = dataset_train.drop(columns='Place')
    y_train = dataset_train['Place']
    X_test = dataset_test.drop(columns='Place')
    y_test = dataset_test['Place']

    # linear regression
    lrm = LRM(X_train, y_train, n_jobs=-1)
    y_pred_lrm = lrm.get_prediction(X_test)
    print(lrm.get_score(X_train, y_train))
    print(lrm.get_score(X_test, y_test))
    print(r2_score(y_test, y_pred_lrm))

    #lrm = LRM(X_train, y_train, solver='newton-cholesky')
    #y_pred_lrm = lrm.get_prediction(X_test)
    #print(lrm.get_score(y_pred_lrm, y_test))