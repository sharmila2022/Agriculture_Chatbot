from flask import Flask, render_template,url_for, request
from flask import jsonify
import random
import numpy
from preprocess import bag_of_words,model, words, labels, data

app=Flask(__name__,template_folder='templates')
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get')
def get_bot_response():
    userText = request.args.get('msg')
    results = model.predict([bag_of_words(userText, words)])[0]
    results_index = numpy.argmax(results)
    tag = labels[results_index]

    for tg in data["intents"]:
        if tg['tag'] == tag:
            responses = tg['responses']

    return jsonify(random.choice(responses))

if __name__ == '__main__':
    app.run(debug=True)
