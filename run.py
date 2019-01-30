import data
import predict
import util

# 사용자 입력값
x = data.getInput()

# 보이스피싱 텍스트(명사만), 단어별 가중치
nouns, vocaWeight = data.getData()

# JARO_WRINKLER
for n in nouns:
    result = predict.jaroWrinkler(util.wordArray(x), util.wordArray(n))

# 커스텀 알고리즘 (구현해야 할 부분)
for n in nouns:
    result = predict.customAlgorithm(util.wordArray(x), util.wordArray(n))
