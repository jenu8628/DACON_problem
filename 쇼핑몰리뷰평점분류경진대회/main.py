import os

import pandas as pd
import re

from matplotlib import pyplot as plt
from konlpy.tag import Mecab
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.layers import Embedding, Dense, LSTM, Bidirectional
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from hanspell import spell_checker
from tqdm import tqdm


def count_missing_value(data):
    print(data.isnull().sum())

def read_stopwords(path):
    stopwords = []
    with open(path, 'r', encoding='utf-8') as f:
        while True:
            line = f.readline()
            if not line: break
            stopwords.append(line.split('\n')[0])
    return stopwords

def preprocess_data(data):
    # data['reviews'] = data['reviews'].str.replace("[^0-9a-zA-Zㄱ-ㅎㅏ-ㅣ가-힣 ]", "")
    # 한글, 영어, 숫자 추출
    result = []
    for sentence in data:
        extract_sentence = "".join(re.findall("[0-9a-zA-Zㄱ-ㅎㅏ-ㅣ가-힣 ]", sentence))
        grammar_sentence = checking_spell(extract_sentence)
        space_sentence = "".join(re.sub(" +", ' ', grammar_sentence))
        result.append([space_sentence])
    # data['reviews'] = data['reviews'].apply(lambda review: "".join(re.findall("[0-9a-zA-Zㄱ-ㅎㅏ-ㅣ가-힣 ]", review)))
    # # 맞춤법 검사
    # data['reviews'] = data['reviews'].apply(lambda x: checking_spell(x))
    # # 공백 제거
    # data['reviews'] = data['reviews'].apply(lambda review: "".join(re.sub(" +", ' ', review)))

    return result

def checking_spell(sentence):
    try:
        spelled_sentence = spell_checker.check(sentence)
        hanspell_sentence = spelled_sentence.checked
    except:
        print(sentence)
        print(type(sentence))
        print(repr(sentence))
        raise
    return hanspell_sentence


def data_tokenization(data, stopwords):
    mecab = Mecab(dicpath=mecab_path)
    result = []
    for sentence in data:
        tokenized_sentence = mecab.morphs(sentence)
        result.append(tokenized_sentence)
        # stopwords_removed_sentence = [word for word in tokenized_sentence if word not in stopwords]
        # result.append(stopwords_removed_sentence)
    return result

def encode_string_to_int(data):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(data)

    minimum_num = 15 # 단어 등장
    total_cnt = len(tokenizer.word_index)   # 전체 단어 수
    rare_cnt = 0    # 등장 빈도수가 minimum_num보다 작은 단어의 개수
    total_freq = 0  # 데이터의 전체 단어 빈도수 총 합
    rare_freq = 0  # 등장 빈도수가 minimum_num보다 작은 단어의 등장 빈도수의 총 합

    for key, value in tokenizer.word_counts.items():
        total_freq += value
        if value < minimum_num:
            rare_cnt += 1
            rare_freq += value

    print('단어 집합(vocabulary)의 전체 크기 :', total_cnt)
    print(f'등장 빈도가 {minimum_num - 1}번 이하인 희귀 단어의 수: {rare_cnt}')
    print("단어 집합에서 희귀 단어의 비율:", (rare_cnt / total_cnt) * 100)
    print("전체 등장 빈도에서 희귀 단어 등장 빈도 비율:", (rare_freq / total_freq) * 100)
    print("------------------------------------------------------------")

    vocab_size = total_cnt - rare_cnt + 1
    tokenizer = Tokenizer(vocab_size)
    tokenizer.fit_on_texts(data)
    return tokenizer, vocab_size

def delete_empty_data(train_input: list, train_output: list):
    input_result = []
    output_result = []
    for idx, data in enumerate(train_input):
        if not data:
            continue
        input_result.append(data)
        output_result.append(train_output[idx])

    return input_result, output_result

def show_sentence_lengh(data):
    max_len = 45
    print('data의 최대 길이 :', max(len(review) for review in data))
    print('data의 평균 길이 :', sum(map(len, data)) / len(data))
    plt.hist([len(review) for review in data], bins=50)
    plt.xlabel('length of samples')
    plt.ylabel('number of samples')
    # plt.show()
    count = 0
    for sentence in data:
        if (len(sentence) <= max_len):
            count = count + 1
    print(f'전체 샘플 중 길이가 {max_len} 이하인 샘플의 비율: {(count/len(data)) * 100}')

def generate_padding(data):
    return pad_sequences(data, maxlen=45, padding='post')

def train_model(train_input, train_output, vocab_size):
    train_output = to_categorical(train_output)
    embedding_dim = 256
    hidden_units = 128
    output_size = 6

    model = Sequential()
    model.add(Embedding(vocab_size, embedding_dim))
    model.add(Bidirectional(LSTM(hidden_units, activation='relu')))
    model.add(Dense(128))
    model.add(Dense(64))
    model.add(Dense(32))
    model.add(Dense(16))
    model.add(Dense(output_size, activation='softmax'))
    print(model.summary())

    early_stop = EarlyStopping(monitor='val_acc', mode='max', verbose=1, patience=5)
    model_checkpoint = ModelCheckpoint(f'best_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)

    model.compile(loss='categorical_crossentropy', metrics=['acc'], optimizer='adam')

    # 훈련 데이터의 20%를 검증 데이터로 분리해서 사용
    # 검증 데이터를 통해서 훈련이 적절히 되고 있는지 확인
    history = model.fit(train_input, train_output, epochs=20, callbacks=[early_stop, model_checkpoint], batch_size=32, validation_split=0.2)
    return model


if __name__ == '__main__':
    # 파일 경로
    base_path = os.path.dirname(os.path.abspath(__file__))
    # 데이터 경로
    train_file_path = os.path.join(base_path, 'dataset', 'train.csv')
    test_file_path = os.path.join(base_path, 'dataset', 'test.csv')
    # Mecab 경로
    mecab_path = os.path.join(base_path, 'mecab', 'mecab-ko-dic')
    # stopwords 경로
    stopwords_path = os.path.join(base_path, 'stopword2.txt')

    # 기본 데이터 읽기
    """
    train : reviews, target
    test : reviews
    """
    train_data = pd.read_csv(train_file_path, encoding='utf-8', index_col=0)
    test_data = pd.read_csv(test_file_path, encoding='utf-8', index_col=0)

    # # 결측치 확인 - 결측치 없음
    # count_missing_value(train_data)
    # count_missing_value(test_data)

    # 데이터 전처리
    train_data = preprocess_data(train_data['reviews'])
    test_data = preprocess_data(test_data['reviews'])

    # 토큰화
    stopwords = read_stopwords(stopwords_path)
    train_input = data_tokenization(train_data, stopwords=stopwords)
    test_input = data_tokenization(test_data, stopwords=stopwords)
    train_output = train_data['target']

    # 정수인코딩
    tokenizer, vocab_size = encode_string_to_int(data=train_input)    # 25000
    train_input = tokenizer.texts_to_sequences(train_input)
    test_input = tokenizer.texts_to_sequences(test_input)

    # 빈 값 제거
    train_input, train_output = delete_empty_data(train_input, train_output)    # 23849

    # 패딩 작업
    show_sentence_lengh(train_input)
    train_input = generate_padding(train_input)
    test_input = generate_padding(test_input)

    # 모델 학습
    model = train_model(train_input, train_output, vocab_size)

    # 예측








