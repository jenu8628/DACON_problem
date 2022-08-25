import sys
import pickle
import pandas as pd
import numpy as np
from scipy.stats import uniform, randint
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.inspection import permutation_importance
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, GradientBoostingClassifier, ExtraTreesClassifier, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_validate, StratifiedKFold, RandomizedSearchCV

# train 데이터 - 훈련 데이터 5497개
# 결측치 없음
# quality(품질)을 제외한 나머지 특성을 가지고 훈련(입력)
# quality가 정답이 됨.(타깃)  3 ~ 9 존재

# test 데이터 - 테스트 데이터 1000개
# 결측치 없음
# train의 콜롬 중 quality(품질)이 사라졌음.
# 분석과 훈련을 통해
# 사라진 quality에 대한 분류를 하면됨.

def read_csv(train_csv):
    # 학습할 훈련 데이터 불러오기
    train = pd.read_csv(train_csv)
    return train

class PreProcessor:
    # 데이터 전처리 클래스
    def __init__(self, train):
        self.train = train

    def object_to_int(self, feature):
        # 오브젝트 타입을 숫자로 변환 하는 함수
        # 현재 type이 오브젝트로 되어있으므로 Red = 1, white = 0로 변경
        enc = LabelEncoder()
        enc.fit(self.train[feature])
        train[feature] = enc.transform(self.train[feature])
        return train

    def feature_data(self, features):
        # 테스트 인풋, 학습 인풋, 학습 타겟 데이터 속성 조절 및 정규화 함수
        # 표준점수 = (값 - 평균) / 분산
        # feature : 속성값 리스트

        train_target = self.train['quality'].copy()
        train_input = self.train.drop(features + ['quality'], axis=1)
        return train_input, train_target

class ModelFactory:
    # 학습 모델 클래스
    def __init__(self, train_input, train_target):
        # train_input : 훈련 입력 데이터
        # train_target : 훈련 정답 데이터
        self.train_input = train_input
        self.train_target = train_target

    def hist_gradient(self):
        # 히스토그램 그래디언트 부스팅 학습 모델
        hgb = HistGradientBoostingClassifier()
        hgb.fit(self.train_input, self.train_target)
        return hgb

    def random_forest(self):
        # 랜덤 포레스트 학습 모델
        rf = RandomForestClassifier()
        rf.fit(self.train_input, self.train_target)
        return rf

    def random_search(self, params):
        # 랜덤 서치 학습 모델
        rs = RandomizedSearchCV(DecisionTreeClassifier(), params)
        rs.fit(self.train_input, self.train_target)
        return rs

    def extra_tree(self):
        # 엑스트라 트리 학습 모델
        et = ExtraTreesClassifier(params)
        et.fit(self.train_input, self.train_target)
        return et

    def gradient_boosting(self):
        # 그래디언트 부스팅 모델
        gb = GradientBoostingClassifier()
        gb.fit(self.train_input, self.train_target)
        return gb

def cross_validation(model, train_input, train_target):
    # 교차 검증 함수
    # model : 데이터 학습 모델
    # train_input : 학습 입력 데이터
    # train_target : 학습 정답 데이터
    return cross_validate(model, train_input, train_target,
                        return_train_score=True,
                        cv=StratifiedKFold(shuffle=True),
                        n_jobs=-1)

def feature_importance(model, train_input, train_target):
    # 속성값 중요도 검사 함수
    # model : 데이터 학습 모델
    # train_input : 학습 입력 데이터
    # train_target : 학습 정답 데이터
    return permutation_importance(model, train_input, train_target,
                                n_repeats=1, n_jobs=-1)

def predict_data(model, test_input):
    # 예측값 반환 함수
    # model : 학습된 모델
    # test_input : 테스트 입력 데이터
    return model.predict(test_input)

def generate_submission(submission_path ,test_target):
    # 모델의 test 예측값 파일 생성 함수
    submission = pd.read_csv(submission_path)
    submission['quality'] = test_target
    submission.to_csv(submission_path, index=False)


if __name__ == '__main__':
    features = [
    'index',
    'fixed acidity',
    'citric acid',
    'residual sugar',
    'free sulfur dioxide',
    'total sulfur dioxide',
    'pH',
    'sulphates',
    ]
    train = read_csv('wine.csv')
    prep = PreProcessor(train)    # 데이터 전처리 클래스
    train= prep.object_to_int('type')
    prep.train = train
    train_input, train_target = prep.feature_data(features)
    lm = ModelFactory(train_input, train_target)
    model = lm.hist_gradient()
    with open('winePickle.pickle', 'wb') as f:
        pickle.dump(model, f)
