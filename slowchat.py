from flask import Flask, request
import logging
import requests

from modules import gpu
from modules import config

logging.basicConfig(level=logging.DEBUG)
app = Flask(__name__)

hostname = gpu.get_hostname()

@app.route('/v1/completions', methods=['POST'])
@app.route('/v1/chat/completions', methods=['POST'])
@app.route('/v1/embeddings', methods=['POST'])
@app.route('/v1/moderations', methods=['POST'])
def intercept_request():
    data = request.json
    model_name = data.get('model', None)
    if not model_name or model_name not in config.AVAILABLE_MODELS:
        return {
            'object': 'error',
            'message': 'Model not specified or not found'
        }, 400
    
    logging.info('Request for model: {}'.format(model_name))
    gpu.load_model(model_name) # Load the model if it's not already loaded
    
    # Forward the request to the model's server
    model = config.AVAILABLE_MODELS[model_name]
    response = requests.post(model['location'] + str(request.path), json=request.json)
    
    logging.debug('Request served! Response: {}'.format(response.text))
    return response.json()

    
@app.route('/v1/models', methods=['GET'])
def get_models():
    return {
        'object': 'list',
        'data': [
            {
                'id': model_name,
                'object': 'model',
                'created': 1686935002,
                'owned_by': hostname
            }
            for model_name in config.AVAILABLE_MODELS.keys()
        ]
    }

@app.route('/v1/models/<model_name>', methods=['GET'])
def get_model(model_name):
    if model_name not in config.AVAILABLE_MODELS:
        return {
            'object': 'error',
            'message': 'Model not found'
        }, 404

    return {
        'object': 'model',
        'id': model_name,
        'created': 1686935002,
        'owned_by': hostname
    }

if __name__ == '__main__':
    # Use Waitress to serve the app
    from waitress import serve
    serve(app, host='0.0.0.0', port=config.PORT)
