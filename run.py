import data
from keras_lstm import train_keras
from keras_lstm import run_keras
import predict
import util

''' 1. 딥러닝 '''
# 케라스 Bi-LSTM 훈련
train_keras.train()
# 케라스 Bi-LSTM 테스트
run_keras.run()

''' 2. 룰 베이스 '''
# 사용자 입력값 ('/data/test.txt')
x = data.getInput()

# 보이스피싱 텍스트(명사만), 단어별 가중치 ('/data/vp.txt', '/data/voca.txt')
nouns, vocaWeight = data.getData()

# JARO-WRINKLER
result = 0
for n in nouns:
    result = max(result, predict.jaroWrinkler(util.wordArray(x), util.wordArray(n)))

# 커스텀 알고리즘
result2 = 0
for n in nouns:
    result2 = max(result2, predict.customAlgorithm(util.wordArray(x), util.wordArray(n)), vocaWeight)
