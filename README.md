# SlowChat

[![GitHub issues](https://img.shields.io/github/issues/Pop101/SlowChat)](https://github.com/Pop101/SlowChat/issues)

Become the greatest gamer known to man

# Table of Contents

- [SlowChat](#slowchat)
- [Table of Contents](#table-of-contents)
- [Overview](#overview)
- [Technologies](#technologies)
- [Getting Started](#getting-started)
  - [Configuration](#configuration)
  - [Installation](#installation)
  - [Usage](#usage)

# Overview

Have an average machine but still want to rival OpenAI? SlowChat spins local LLM models for you, balancing VRAM usage to ensure the most number of users can be served

# Technologies

This project is created with:

- [Flask](https://flask.palletsprojects.com/en/3.0.x/): 3.0.3
- [waitress](https://docs.pylonsproject.org/projects/waitress/en/stable/): 3.0.0
- [aiohttp](https://docs.aiohttp.org/en/stable/): 3.10.6
- [ortools](https://developers.google.com/optimization): 9.11.4210

# Getting Started

This project was created as a replacement for [FastChat](https://github.com/lm-sys/FastChat), which uses llm-servers to serve a variety of models simultaneously off a single endpoint. This project is designed for systems that cannot simultaneously load multiple LLMs, and instead spins down and up new models as needed.

## Configuration

Ensure you have a way to serve individual LLMs. My preferred method is [vLLM](https://github.com/vllm-project/vllm). Write all your desired models to `./config.json`, in the following form:

```json
"models": [
    {
        "name": "MODELNAME",
        "location": "localhost:3001",
        "load_command": "vllm serve blahblahblah"
    }
]
```

Note that the `name` must be the *exact* name of the model, expected by vLLM. This name will be directly passed through to the underlying location endpoint.

## Installation

Clone the Repo and ensure poetry is installed
```sh
git clone https://github.com/Pop101/SlowChat
pip install poetry
```

Install the dependencies using poetry's version management
```sh
poetry install
```

Now you can serve the applet itself.
```sh
poetry run python ./slowchat.py
```

## Usage

This project can be used as an OpenAI API replacement, with the following endpoints:

**Managed by SlowChat**: \
- `GET /v1/models` - Returns a list of all models available to the user
- `GET /v1/models/{model}` - Returns the details of a specific model

**Passed to Endpoints**: \
- `POST /v1/completions`
- `POST /v1/chat/completions`
- `POST /v1/embeddings`
- `POST /v1/moderations`

If your endpoint does not support embeddings or moderations, it will return the exact error message from the underlying endpoint. This is expected behaviour and is entirely okay.