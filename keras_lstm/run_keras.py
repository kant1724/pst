from keras.preprocessing import sequence
from keras_lstm import preprocess
from keras.models import model_from_json
import data
maxlen = 500

def run():
    # 테스트 데이터 받기
    test_arr = data.getKerasTestData()

    # 테스트 데이터 숫자 토큰으로 가공
    x_test = preprocess.process_test_data(test_arr)

    # 테스트 데이터 패딩처리
    x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

    # 케라스 설정값 초기화
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()

    # 저장된 케라스 모델 로드
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("model.h5")
    loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    # 모델로 예측
    score = loaded_model.predict(x_test, verbose=0)

    # 점수
    print(score)