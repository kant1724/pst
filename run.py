import data
import predict
import util

# 사용자 입력값 ('/data/input.txt')
x = data.getInput()

# 보이스피싱 텍스트(명사만), 단어별 가중치 ('/data/vp_text.txt', '/data/voca_data.txt')
nouns, vocaWeight = data.getData()

# JARO_WRINKLER
result = 0
for n in nouns:
    result = max(result, predict.jaroWrinkler(util.wordArray(x), util.wordArray(n)))

# 커스텀 알고리즘 (구현해야 할 부분)
result2 = 0
for n in nouns:
    result2 = max(result2, predict.customAlgorithm(util.wordArray(x), util.wordArray(n)))
