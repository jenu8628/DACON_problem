import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re
import pickle

from sklearn.metrics import mean_squared_error
from sklearn.inspection import permutation_importance
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import cross_validate, KFold
from xgboost import XGBRegressor, XGBRFRegressor



def read_csv(train_csv):
    return pd.read_csv(train_csv).drop(['title', 'director', 'release_time'], axis=1)

class DataAnalyzer:
    def __init__(self, train):
        self.train = train

    plt.figure(figsize=(10, 10))
    # 분포도
    def hist(self):
        self.train.hist(bins=100)
        plt.show()

    # 상관관계
    def correlation(self):
        sns.heatmap(data=self.train.corr(), annot=True)
        plt.show()

    # 각 피쳐와 관객수 사이의 분포 그래프
    def distribute_between(self, feature):
        for i in range(0, 7):
            plt.subplot(3, 3, i+1)
            sns.barplot(x=feature, y=self.train.columns[i], data=self.train)
        plt.tight_layout()
        plt.show()

def get_dis(x):
    if 'CJ' in x or 'CGV' in x:
        return 'CJ'
    elif '쇼박스' in x:
        return '쇼박스'
    elif 'SK' in x:
        return 'SK'
    elif '리틀빅픽' in x:
        return '리틀빅픽처스'
    elif '스폰지' in x:
        return '스폰지'
    elif '싸이더스' in x:
        return '싸이더스'
    elif '에이원' in x:
        return '에이원'
    elif '마인스' in x:
        return '마인스'
    elif '마운틴픽' in x:
        return '마운틴픽처스'
    elif '디씨드' in x:
        return '디씨드'
    elif '드림팩트' in x:
        return '드림팩트'
    elif '메가박스' in x:
        return '메가박스'
    elif '마운틴' in x:
        return '마운틴'
    else:
        return x


class PreProcessor:
    def __init__(self, train):
        self.train = train


    # 영화 배급사 전처리
    def pre_distributor(self):
        train['distributor'] = self.train.distributor.str.replace("(주)", '')
        train['distributor'] = [re.sub(r'[^0-9a-zA-Z가-힣]', '', x) for x in train.distributor]
        train['distributor'] = train.distributor.apply(get_dis)
        return train


    # 오브젝트 -> 인트형 전환
    def object_to_int(self):
        # 컬럼의 카테고리 별로 box_off_num의 평균이 낮은 순서부터 책정했음.
        train['genre_rank'] = self.train.genre.map({'뮤지컬': 1, '다큐멘터리': 2, '서스펜스': 3, '애니메이션': 4, '멜로/로맨스': 5,
                                            '미스터리': 6, '공포': 7, '드라마': 8, '코미디': 9, 'SF': 10, '액션': 11, '느와르': 12})
        # num_rank 생성
        # 배급사별 영화 관객수 중위값 기준으로 배급사 랭크 인코딩
        tr_nm_rank = self.train.groupby('distributor').box_off_num.median().reset_index(name='num_rank').sort_values(by='num_rank')
        tr_nm_rank['num_rank'] = [i + 1 for i in range(tr_nm_rank.shape[0])]
        train_data = pd.merge(train, tr_nm_rank, how='left')
        return train_data

    def feature_data(self):
        train_target = np.log1p(self.train.box_off_num)
        train_input = self.train[['num_rank', 'time', 'num_staff', 'num_actor', 'genre_rank', 'screening_rat']]
        train_input = pd.get_dummies(columns=['screening_rat'], data=train_input)
        train_input['num_actor'] = np.log1p(train_input['num_actor'])
        return train_input, train_target

# 학습 모델 클래스
class ModelFactory:

    def __init__(self, train_input, train_target):
        self.train_input = train_input
        self.train_target = train_target

    # 히스토그램 그래디언트 부스팅 학습 모델
    def hist_gradient(self):
        hg = HistGradientBoostingRegressor()
        return hg.fit(self.train_input, self.train_target)

    # 랜덤 포레스트 학습 모델
    def random_forest(self):
        rf = RandomForestRegressor()
        return rf.fit(self.train_input, self.train_target)

    # LightGBM 모델
    def light_gbm(self):
        lgbm = LGBMRegressor()
        return lgbm.fit(self.train_input, self.train_target)

    def xgbm(self):
        xg = XGBRFRegressor()
        return xg.fit(self.train_input, self.train_target)

    def extra(self):
        extra = ExtraTreesRegressor()
        return extra.fit(self.train_input, self.train_target)

    def gradientboosting(self):
        gb = GradientBoostingRegressor()
        return gb.fit(self.train_input, self.train_target)

# 예측값 반환 함수
def predict_data(model, test_input):
    # model : 학습된 모델
    # test_input : 테스트 입력 데이터
    return np.expm1(model.predict(test_input))

if __name__ == "__main__":
    train = read_csv('movies.csv')

    # 1. 데이터 분석 및 전처리
    pre = PreProcessor(train)
    # 1-1) 영화 배급사 전처리
    train = pre.pre_distributor()
    pre.train= train
    dist = train.groupby('distributor').box_off_num.median().reset_index(name='num_rank').sort_values(by='num_rank')
    arr = np.array(dist['distributor'])
    # 1-2) 오브젝트형을 int형으로 전환하자.
    # 장르는 장르별 영화 관객수 평균값으로 작은 값부터 넣어주고
    # 배급사는 배급사별 영화 관객수 중위값 기준으로 하자.
    train = pre.object_to_int()
    pre.train = train

    train_input, train_target = pre.feature_data()

    # e표현 없애기
    np.set_printoptions(precision=6, suppress=True)
    # 2. 모델링
    lr = ModelFactory(train_input, train_target)
    gb_model = lr.gradientboosting()
    genre_dict = {'뮤지컬': 1, '다큐멘터리': 2, '서스펜스': 3, 
                '애니메이션': 4, '멜로/로맨스': 5,'미스터리': 6, 
                '공포': 7, '드라마': 8, '코미디': 9, 
                'SF': 10, '액션': 11, '느와르': 12}
    with open('movies.pickle', 'wb') as f:
        pickle.dump(gb_model, f)
        pickle.dump(arr, f)
        pickle.dump(genre_dict, f)

