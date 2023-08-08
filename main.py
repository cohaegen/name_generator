from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import tensorflow as tf
import numpy as np

TIMESERIES_CONTEXT = 32
INPUT_FILENAME = 'shakespeare.tflite'
MAX_NAME_LENGTH = 32

app = FastAPI()


class NameOptions(BaseModel):
    """
    Enable options when generating a name
    """
    starts_with: str = ''
    max_length: int = MAX_NAME_LENGTH


def generate_tflite_text(interpreter: tf.lite.Interpreter, context='\n  ', num_chars=100, stop_char=None):
    """
    Generates text from a TF Lite interpreter (created from a TF Lite trained model)
    You can pass it context which is used as the "what text came already" when prompted the model.
    The model fills in "unknown" tokens to pad up to the maximum context length.
    The model will predict what characters come next.
    You can request a certain number of characters and a stop character which will stop generation.
    """
    signature = interpreter.get_signature_runner()
    for _ in range(num_chars):
        inf_context = ['[UNK]']*(TIMESERIES_CONTEXT - len(context)) + list(context)  # pad
        inf_context = inf_context[-TIMESERIES_CONTEXT:]  # truncate
        next_char = signature(values_input=np.array([list(inf_context)]))['choose_character_layer'][0,0].decode()
        if stop_char and next_char == stop_char:
            break
        context += next_char
    return context

# Read a TF Lite interpreter from disk
interpreter = tf.lite.Interpreter(model_path=INPUT_FILENAME)

# Define API endpoints
@app.get('/', response_class=HTMLResponse)
async def root():
    """
    Return an index page
    """
    return open('static/index.html').read()

@app.get('/name')
async def get_name():
    """
    Generate and return a single Shakearean name
    """
    # Prompting with "\n  " as the context will encourage it to create a new character name,
    # since in Shakespeare's plays, a new character speaking is preceeded by a newline and two spaces
    # The model already has a layer that only generates upper case characters and the period. Character
    # names are all upper case and end with a period.
    return {'name': generate_tflite_text(interpreter, context='\n  ', num_chars=50, stop_char='.')[3:]}

@app.post('/name')
async def post_name(name_options: NameOptions):
    """
    Generate a Shakespearean character name with some options
    """
    context = '\n  ' + name_options.starts_with
    num_chars = min(MAX_NAME_LENGTH, name_options.max_length) - len(name_options.starts_with)
    return {'name': generate_tflite_text(interpreter, context=context, num_chars=num_chars, stop_char='.')[3:]}
