import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge


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


class RidgeModel(Model):

    def __init__(self, X, y, description='ridge regression model', **kwargs):
        Model.__init__(self, description,
                       model=Ridge(**kwargs).fit(X, y))


class RFR(Model):

    def __init__(
            self,
            X,
            y,
            description='random forest regression model',
            **kwargs):
        Model.__init__(self, description,
                       model=RandomForestRegressor(**kwargs).fit(X, y))


class GBR(Model):

    def __init__(
            self,
            X,
            y,
            description='gradient boosting regression model',
            **kwargs):
        Model.__init__(self, description,
                       model=GradientBoostingRegressor(**kwargs).fit(X, y))


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

    # ridge
    rm = RidgeModel(X_train, y_train, alpha=2.0)
    y_pred_rm = rm.get_prediction(X_test)
    print(rm.get_score(X_train, y_train))
    print(rm.get_score(X_test, y_test))

    # random forest
    rfr = RFR(
        X_train,
        y_train,
        max_depth=10,
        max_features=None,
        min_samples_split=5,
        min_samples_leaf=2,
        n_estimators=100)
    y_pred_rfr = rfr.get_prediction(X_test)
    print(rfr.get_score(X_train, y_train))
    print(rfr.get_score(X_test, y_test))

    # gradient boosting
    gbr = GBR(
        X_train,
        y_train,
        learning_rate=0.01,
        max_depth=3,
        max_features=None,
        min_samples_leaf=2,
        min_samples_split=10,
        n_estimators=500,
        subsample=0.8)
    y_pred_gbr = gbr.get_prediction(X_test)
    print(gbr.get_score(X_train, y_train))
    print(gbr.get_score(X_test, y_test))
