import json

raw_cfg = json.loads(open('config.json').read())

PORT = raw_cfg['port']

AVAILABLE_MODELS = dict()
for model in raw_cfg.get('models', {}):
    AVAILABLE_MODELS[model['name']] = model