from keras.preprocessing import sequence
from keras_lstm import preprocess
from keras.models import model_from_json
import data
maxlen = 500

def run():
    test_arr = data.getKerasTestData()
    x_test = preprocess.process_test_data(test_arr)
    x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("model.h5")

    loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    score = loaded_model.predict(x_test, verbose=0)
    print(score)