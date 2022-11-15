import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from ngboost import NGBRegressor

def read_csv(train, test):
    return pd.read_csv(train), pd.read_csv(test)


# 전처리 클래스
class PreProcessor:
    def __init__(self, train, test):
        self.train = train
        self.test = test

    # continent : 선수들의 국적이 포함되어 있는 대륙
    # 인트형으로 변환
    def conversion_continent(self):
        self.test['continent'] = self.test['continent'].map({'oceania': 1, 'asia': 2, 'africa': 3, 'south america': 4, 'europe': 5})
        self.train['continent'] = self.train['continent'].map({'oceania': 1, 'asia': 2, 'africa': 3, 'south america': 4, 'europe': 5})
        return self.train, self.test

    # contract_until : 이적 시기
    # 인트형으로 변환
    def conversion_contract_until(self):
        def dis(x):
            if x == 'Dec 31, 2018':
                return '2019'
            elif x == 'Jun 30, 2020':
                return '2020.5'
            elif x == 'Jun 30, 2019':
                return '2019.5'
            elif x == 'May 31, 2020':
                return '2020.3333'
            elif x == 'May 31, 2019':
                return '2019.3333'
            elif x == 'Jan 31, 2019':
                return '2019.0833'
            elif x == 'Jan 1, 2019':
                return '2019'
            elif x == 'Jan 12, 2019':
                return '2019.034'
            elif x == 'Dec 31, 2019':
                return '2020'
            elif x == 'Jun 1, 2019':
                return '2019.416'
            else:
                return x
        self.train['contract_until'] = self.train.contract_until.apply(dis).astype('float64') - 2018
        self.test['contract_until'] = self.test.contract_until.apply(dis).astype('float64') - 2018
        return self.train, self.test

    # train_input, train_target, test_input 데이터 만들기
    def feature_data(self, feature):
        train_target = np.log1p(self.train['value'])
        train_input = self.train[feature]
        test_input = self.test[feature]
        # 더미 데이터 생성
        train_input = pd.get_dummies(columns=['continent', 'position'], data=train_input)
        test_input = pd.get_dummies(columns=['continent', 'position'], data=test_input)
        return train_input, train_target, test_input



# 데이터 분석 클래스
class DataAnalyzer:
    def __init__(self, train):
        self.train = train

    plt.figure(figsize=(10, 10))
    # 분포도
    def hist(self):
        self.train.hist(bins=100)
        plt.show()

    # 상관관계
    def correlation(self, *feature):
        if feature:
            sns.heatmap(data=self.train.corr()[[feature[0]]], annot=True)
        else:
            sns.heatmap(data=self.train.corr(), annot=True)
        plt.show()

    # 각 피쳐와 관객수 사이의 분포 그래프
    def distribute_between(self, feature, *features):
        for i in range(len(features[0])):
            plt.subplot(3, 3, i+1)
            plt.scatter(self.train[features[0][i]], self.train[feature], alpha=0.2)
            plt.ylabel(feature)
            plt.xlabel(features[0][i])
            # sns.histplot(x=self.train[features[0][i]], y=self.train[feature])

            # sns.stripplot(x=self.train[features[0][i]], y=self.train[feature])
        plt.tight_layout()
        plt.show()


class ModelFactory:
    def __init__(self, train_input, train_target, test_input):
        self.train_input = train_input
        self.train_target = train_target
        self.test_input = test_input

    def ngboost(self):
        kf = KFold(n_splits=10, random_state=521, shuffle=True)
        ngb = NGBRegressor(random_state=521, verbose=500, n_estimators=500)
        ngb_pred = np.zeros((self.test_input.shape[0]))
        rmse_list = []
        for tr_idx, val_idx in kf.split(self.train_input, self.train_target):
            tr_x, tr_y = self.train_input.iloc[tr_idx], self.train_target.iloc[tr_idx]
            val_x, val_y = self.train_input.iloc[val_idx], self.train_target.iloc[val_idx]

            ngb.fit(tr_x, tr_y)
            pred = np.expm1([0 if x < 0 else x for x in ngb.predict(val_x)])

            rmse = np.sqrt(mean_squared_error(np.expm1(val_y), pred))
            rmse_list.append(rmse)

            sub_pred = np.expm1([0 if x < 0 else x for x in ngb.predict(self.test_input)]) / 10
            ngb_pred += sub_pred
        print(f'{ngb.__class__.__name__}의 10fold 평균 RMSE는 {np.mean(rmse_list)}')
        return ngb_pred, rmse_list, ngb

def generate_submission(submission_path ,test_target, save_path):
    submission = pd.read_csv(submission_path)
    submission['value'] = test_target
    submission.to_csv(save_path, index=False)

if __name__ == "__main__":
    train, test = read_csv('FIFA_train.csv', 'FIFA_test.csv')
    # e표현 없애기
    pd.options.display.float_format = '{:.5f}'.format
    np.set_printoptions(precision=6, suppress=True)

    # 1. 데이터 전처리
    pre = PreProcessor(train, test)
    # 1-1) contract_until을 인트형으로 변환
    train, test = pre.conversion_contract_until()
    pre.train, pre.test = train, test
    print(train['continent'].value_counts())
    # 1-2) log Transformation
    train[['age', 'stat_potential']] = np.log1p(train[['age', 'stat_potential']])
    test[['age', 'stat_potential']] = np.log1p(test[['age', 'stat_potential']])
    # 1-5) train_input, train_target, test_input 데이터 생성
    temp = [
        'age',
        'continent',
        'contract_until',
        'position',
        # 'prefer_foot',
        'reputation',
        'stat_overall',
        'stat_potential',
        'stat_skill_moves'
    ]
    train_input, train_target, test_input = pre.feature_data(temp)
    print(train_input.info())
    print(train_target.head())
    print(np.expm1(train_target.head()))
    # 모델링 및 예측
    lr = ModelFactory(train_input, train_target, test_input)
    ngb_pred, rmse_list, model = lr.ngboost()
    print(ngb_pred)
    print(rmse_list)
    print(np.expm1(model.predict(test_input)))
    #
    # # 5. test_input을 파일로 만들기
    # generate_submission('submission.csv', ngb_pred, 'predict.csv')

