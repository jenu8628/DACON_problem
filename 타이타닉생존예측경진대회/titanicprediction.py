import sys
import pickle

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from lightgbm import LGBMClassifier

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, GradientBoostingClassifier, ExtraTreesClassifier, RandomForestClassifier
from sklearn.model_selection import cross_validate, StratifiedKFold, RandomizedSearchCV
from sklearn.inspection import permutation_importance

# 데이터 불러오기 함수
def read_csv(train_csv, test_csv):
    return pd.read_csv(train_csv), pd.read_csv(test_csv)


# 데이터 분석, 그래프 클래스
class DataAnalyzer:
    def __init__(self, train, test):
        self.train = train
        self.test = test

    # 상관관계 그래프 생성
    def correlation(self):
        plt.figure(figsize=(12, 12))
        sns.heatmap(data=self.train.corr(), annot=True)
        plt.show()

# 전처리 클래스
class PreProcessor:
    def __init__(self, train):
        self.train = train

    # object를 int형으로 바꾼다.
    def object_to_int(self, *features):
        # female: 0, male: 1
        # C: 0, Q: 1, S: 2
        enc = LabelEncoder()
        for feature in features:
            enc.fit(self.train[feature])
            train[feature] = enc.transform(self.train[feature])
        return train

    # 결측치 처리 : 평균으로 결측치 채우기
    def fill_value(self):
        train = self.train.fillna(round(self.train.mean()))
        return train

    # 결측치처리
    # Age : Pclass에 따른 평균으로 채운다.
    # Embarked 다수인 S로 채운다.
    # test의 Fare: Pclass 3의 평균인 12로 채운다.
    def Age_mean(self):
        # 결측값을 갖는 df 생성
        train_age_null = self.train[self.train.Age.isnull()]

        # 데이터를 1, 2, 3으로 나눈다.
        train_firstclass = train_age_null[train_age_null.Pclass == 1]
        train_secondclass = train_age_null[train_age_null.Pclass == 2]
        train_thirdclass = train_age_null[train_age_null.Pclass == 3]

        # 각 테이블에 결측치를 채운다.
        train_firstclass = train_firstclass.fillna(value='38')
        train_secondclass = train_secondclass.fillna(value='30')
        train_thirdclass = train_thirdclass.fillna(value='25')

        # train의 Age의 결측값을 포함하는 행을 삭제
        train_drop = self.train.dropna(subset=['Age'])
        train = pd.concat([train_drop, train_firstclass, train_secondclass, train_thirdclass])
        # 인트형 전환
        train = train.astype({'Age': 'int'})
        # # Embarked, Fare : 결측치 채우기
        # train["Embarked"].fillna('S', inplace=True)
        # Embarked, Fare : 결측치 삭제
        train = train.dropna(subset=['Embarked'])

        # SibSp와 Parch를 더해서 가족의 수 컬럼을 새로 만든다.
        train['Family_Size'] = train['SibSp'] + train['Parch']
        # survibed의 내용에 따라 그룹으로 묶는다.
        train['Family_Size'].apply(lambda x: 0 if x == 0 else x)
        train['Family_Size'].apply(lambda x: 0 if 0 < x <= 3 else x)
        train['Family_Size'].apply(lambda x: 0 if 3 < x else x)
        # Fare는 표준편차가 너무 크므로 Log를 씌워주자.
        train['Fare'] = np.log1p(train['Fare'])
        return train

    # 필요없는 속성값 제거 후 입력과 타깃 데이터 나누기
    def feature_data(self, features):
        train_target = self.train['Survived'].copy()
        train_input = self.train.drop(features + ['Survived'], axis=1)
        return train_input, train_target


# 학습 모델 클래스
class ModelFactory:
    def __init__(self, train_input, train_target):
        # train_input : 훈련 입력 데이터
        # train_target : 훈련 정답 데이터
        self.train_input = train_input
        self.train_target = train_target

    # 히스토그램 그래디언트 부스팅 학습 모델
    def hist_gradient(self):
        hgb = HistGradientBoostingClassifier()
        hgb.fit(self.train_input, self.train_target)
        return hgb

    # 랜덤 포레스트 학습 모델
    def random_forest(self):
        rf = RandomForestClassifier()
        rf.fit(self.train_input, self.train_target)
        return rf

    # 랜덤 서치 학습 모델
    def random_search(self, params):
        rs = RandomizedSearchCV(DecisionTreeClassifier(), params)
        rs.fit(self.train_input, self.train_target)
        return rs

    # 엑스트라 트리 학습 모델
    def extra_tree(self):
        et = ExtraTreesClassifier()
        et.fit(self.train_input, self.train_target)
        return et

    # 그래디언트 부스팅 모델
    def gradient_boosting(self):
        gb = GradientBoostingClassifier()
        gb.fit(self.train_input, self.train_target)
        return gb

    # LightGBM 모델
    def light_gbm(self):
        lgbm = LGBMClassifier()
        lgbm.fit(self.train_input, self.train_target)
        return lgbm

    # 로지스틱 회귀
    def logistic_reg(self):
        lr = LogisticRegression()
        lr.fit(self.train_input, self.train_target)
        return lr


# 교차 검증 함수
def cross_validation(model, train_input, train_target):
    # model : 데이터 학습 모델
    # train_input : 학습 입력 데이터
    # train_target : 학습 정답 데이터
    return cross_validate(model, train_input, train_target,
                        return_train_score=True,
                        cv=StratifiedKFold(shuffle=True),
                        n_jobs=-1,
                        scoring='accuracy')

# 속성값 중요도 검사 함수
def feature_importance(model, train_input, train_target):
    # model : 데이터 학습 모델
    # train_input : 학습 입력 데이터
    # train_target : 학습 정답 데이터
    return permutation_importance(model, train_input, train_target,
                                n_repeats=1, n_jobs=-1)

# 예측값 반환 함수
def predict_data(model, test_input):
    # model : 학습된 모델
    # test_input : 테스트 입력 데이터
    return model.predict(test_input)

# 모델의 test 예측값 파일 생성 함수
def generate_submission(submission_path ,test_target, save_path):
    submission = pd.read_csv(submission_path)
    submission['Survived'] = test_target
    submission.to_csv(save_path, index=False)


if __name__ == "__main__":
    train = pd.read_csv('titanic.csv')
    train = train.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
    pre = PreProcessor(train)
    train = pre.Age_mean()
    pre.train = train
    train = pre.object_to_int('Sex', 'Embarked')
    pre.train = train
    features = [
        'SibSp',
        'Parch',
        'Fare',
        'Embarked',
        # 'Sex',
        # 'Pclass',
        # 'Age',
        # 'Family_Size'
    ]
    train_input, train_target = pre.feature_data(features)
    lm = ModelFactory(train_input, train_target)
    model = lm.hist_gradient()
    with open('titanicPickle.pickle', 'wb') as f:
        pickle.dump(model, f)


else:
    train = pd.read_csv('titanic.csv')
    train = train.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
    pre = PreProcessor(train)
    train = pre.Age_mean()
    pre.train = train
    train = pre.object_to_int('Sex', 'Embarked')
    pre.train = train
    features = [
        'SibSp',
        'Parch',
        'Fare',
        'Embarked',
        # 'Sex',
        # 'Pclass',
        # 'Age',
        # 'Family_Size'
    ]
    train_input, train_target = pre.feature_data(features)
    lm = ModelFactory(train_input, train_target)
    model = lm.hist_gradient()