# Shakespeare Name Generator

This is a character name generator that uses machine learning trained on The Complete Works of William Shakespeare. The names it generates are useful for games, stories, role playing, probably other things. It uses a Generative Pre-Trained Transformer (GPT) similar to large language models but on a much smaller scale. The name generator prompts the GPT into thinking it's generating a character name by prompting the GPT with a new line and two spaces, then generating ASCII until it generates a period. This is always how a new character starts speaking in the version of Shakespeare this GPT is trained on.

## Quick Start

```
pip3 install -r requirements.txt
uvicorn main:app
```

Now you can visit http://localhost:8000

## Files:

* `shakespeare.tflite` is a trained model that takes characters as context for the input, and outputs a character that it predicts will be the next one. It is limited to picking upper case ASCII and a period.
* `train_name_generator.py` will train a new model. This shouldn't be necessary as you can use `shakespeare.tflite`, but if you want to change any parameters then you can train a new model.
* `main.py` is a fastAPI web server that creates an API for generating character names. fastAPI will automatically create API documentation at http://localhost:8000/docs