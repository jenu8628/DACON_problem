import pandas as pd
import numpy as np
import pickle
import dill
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from ngboost import NGBRegressor

# 전처리 클래스
class PreProcessor:
    def __init__(self, train):
        self.train = train


    # continent : 선수들의 국적이 포함되어 있는 대륙
    # 인트형으로 변환
    def conversion_continent(self):
        self.train['continent'] = self.train['continent'].map({'oceania': 1, 'asia': 2, 'africa': 3, 'south america': 4, 'europe': 5})
        return self.train

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
        return self.train

    # train_input, train_target, test_input 데이터 만들기
    def feature_data(self, feature):
        train_target = np.log1p(self.train['value'])
        train_input = self.train[feature]
        # 더미 데이터 생성
        train_input = pd.get_dummies(columns=['continent', 'position'], data=train_input)
        return train_input, train_target


class ModelFactory:
    def __init__(self, train_input, train_target):
        self.train_input = train_input
        self.train_target = train_target

    def ngboost(self):
        ngb = NGBRegressor(random_state=521, verbose=500, n_estimators=500)
        kf = KFold(n_splits=10, random_state=521, shuffle=True)
        # ngb_pred = np.zeros((feature[0].shape[0]))
        rmse_list = []
        for tr_idx, val_idx in kf.split(self.train_input, self.train_target):
            tr_x, tr_y = self.train_input.iloc[tr_idx], self.train_target.iloc[tr_idx]
            # val_x, val_y = self.train_input.iloc[val_idx], self.train_target.iloc[val_idx]

            ngb.fit(tr_x, tr_y)
            # pred = np.expm1([0 if x < 0 else x for x in ngb.predict(val_x)])

            # rmse = np.sqrt(mean_squared_error(np.expm1(val_y), pred))
            # rmse_list.append(rmse)

            # sub_pred = np.expm1([0 if x < 0 else x for x in ngb.predict(feature[0])]) / 10
            # ngb_pred += sub_pred
        return ngb



if __name__ == "__main__":
    train = pd.read_csv('fifa.csv')
    # e표현 없애기
    pd.options.display.float_format = '{:.5f}'.format
    np.set_printoptions(precision=6, suppress=True)

    # 1. 데이터 전처리
    pre = PreProcessor(train)
    # 1-1) contract_until을 인트형으로 변환
    train = pre.conversion_contract_until()

    # 1-2) log Transformation
    train[['age', 'stat_potential']] = np.log1p(train[['age', 'stat_potential']])

    # 1-3) train_input, train_target 데이터 생성
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
    train_input, train_target = pre.feature_data(temp)

    # 모델링 및 예측
    lr = ModelFactory(train_input, train_target)
    model = lr.ngboost()
    with open('fifa.pickle', 'wb') as f:
        dill.dump(model, f)

    import re
    package_list = {}
    for i in list(locals().keys()) :
        if re.search("^_|ipython" , i) is None :
            if re.match("In|Out|exit|quit|i" , i ) is None :
                
                if re.search("^<function|^<module|^<class" , str(locals()[i])) is not None :
                    package_list[i] = locals()[i]
                    print(package_list[i], locals()[i])


