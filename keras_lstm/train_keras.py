from keras.preprocessing import sequence
from keras import layers
from keras.models import Sequential
from keras_lstm import preprocess
import data
# 특성으로 사용할 단어의 수
max_features = 10000
# 사용할 텍스트의 길이
maxlen = 500

def train():
    # 훈련 데이터 전처리
    preprocess.prepare_custom_data()

    # 훈련 데이터 초기화
    x_train, y_train = data.getKerasTrainingData()

    # 훈련 데이터 패딩처리
    x_train = sequence.pad_sequences(x_train, maxlen=maxlen)

    # 케라스 모델정의
    model = Sequential()
    model.add(layers.Embedding(max_features, 32))
    model.add(layers.Bidirectional(layers.LSTM(32)))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])

    # 훈련
    model.fit(x_train, y_train, epochs=5, batch_size=128, validation_split=0.2)

    # 모델설정 저장
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)

    # 모델 저장
    model.save_weights("model.h5")