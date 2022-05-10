from flask import Flask, request, jsonify
import logging

from malaya_ner_functions import text_entities


logging.basicConfig(filename='logs.txt',
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.DEBUG)

logging.info("Starting Malaya NER API")


app = Flask(__name__)


@app.route('/text_entities', methods=['GET', 'POST'])
def get_text_entities():

    content = request.get_json(force=True)

    text = str(content['text'])
    logging.info("Processing following text: " + text)
    detected_entities = text_entities(text)

    return jsonify(detected_entities)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080)
