"""
Generate character names using a Generative Pre-Trained Transformer trained on
The Complete Works of William Shakespeare

Run a web service to let people generate names
"""

import argparse
import string
import logging
import tensorflow as tf
import numpy as np

INPUT_FILE = 't8.shakespeare.txt'
TIMESERIES_CONTEXT = 32
NUM_HEADS = 4
HEAD_SIZE = 32
DROPOUT = 0.2
NUM_LAYERS = 6  # Number of latent self-attention layers
LEARNING_RATE = 3e-4
BATCH_SIZE = 32


class TokenAndPositionEmbedding(tf.keras.layers.Layer):
    """
    Create embeddings for both tokens and positions (added together)
    """
    def __init__(self, vocabulary_size: int, timeseries_size: int, embedding_size: int):
        """
        Initialize the layer
        @param vocabulary_size[in]: number of possible tokens
        @param timeseries_size[in]: number of elements in the time dimension
        @param embedding_size[in]: how many units to use in the embedding output dimension

        The position embedding is a simple incrementing series
        """
        super().__init__()
        self.positions = tf.range(timeseries_size)
        self.tok_embedding = tf.keras.layers.Embedding(vocabulary_size, embedding_size)
        self.pos_embedding = tf.keras.layers.Embedding(timeseries_size, embedding_size)
    
    def call(self, values):
        """
        Pass value tokens through the embedding and add a position embedding
        """
        return self.tok_embedding(values) + self.pos_embedding(self.positions)


class TransformerSelfAttention(tf.keras.layers.Layer):
    """
    Transformer self-attention layer
    This is an implementation of the Transformer Decoder from "Attention Is All You Need (Visnay, 2021)"
    with the layer normalization before the layer instead of after and a GeLU activation.

    It instantiates a Keras MultiHeadAttention layer, then passes that to a feed forward block.

    The input tensors to this layer should be three dimensional: (batch, index/time, channels)
    The number of channels should equal num_heads*head_size
    The output tensor shape matches the input shape (batch, index/time, channels)
    """
    def __init__(self, num_heads: int, head_size: int, dropout: int):
        """
        Initialize the Transformer layer.
        @param num_heads[in]: number of attention heads. Each of the heads can attend to the entire input.
        @param head_size[in]: number of elements for each head
        @param dropout[in]: dropout regularization (range from 0.0 to 1.0) - randomly cuts this proportion of 
                            connections in the attention and feed forward blocks during training
        """
        super().__init__()
        self.attention = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=head_size, dropout=dropout)
        self.feed_forward = tf.keras.Sequential([tf.keras.layers.Dense(4*num_heads*head_size, activation='gelu'),
                                                 tf.keras.layers.Dense(num_heads*head_size),
                                                 tf.keras.layers.Dropout(dropout)])
        self.layer_norm1 = tf.keras.layers.LayerNormalization()
        self.layer_norm2 = tf.keras.layers.LayerNormalization()
    
    def call(self, values):
        """
        Perform self-attention and feed forward on values
        values should be 3-dimensional: (batch, index/time, channels).
        Number of channels should equal num_heads*head_size
        """
        norm_values = self.layer_norm1(values)
        attn = values + self.attention(norm_values, norm_values, use_causal_mask=True)
        norm_attn = self.layer_norm2(attn)
        feed_fwd = attn + self.feed_forward(norm_attn)
        return feed_fwd


def generate_batch(data: np.array, context: int, batch_size: int):
    """
    Return a generator that creates one batch of training data at a time. The generator
    returns tuples of (values, targets) which are inputs and ouputs for training an ML model.
    It can be used with the Keras Model.fit() method.

    @param data[in]: A one-dimensional array of input data. It should not be shuffled.
    @param context[in]: number of data elements to include in the index/time dimension
    @param batch_size[in]: number of examples to put in each batch
    """
    starting_offsets = np.arange(len(data)-context)
    np.random.shuffle(starting_offsets)
    for batch_idx in range(0, len(starting_offsets), batch_size):
        batch_starting_offsets = starting_offsets[batch_idx:(batch_idx+batch_size), np.newaxis]
        batch_indices = batch_starting_offsets + np.arange(context)
        values = data[batch_indices]
        targets = data[batch_indices+1]
        yield values, targets

def read_input(input_file: str):
    """
    Read text input from a given filename and encode the ASCII characters with a Keras StringLookup layer.
    Returns a tuple of encoded character values and the encoder.
    """
    raw_input = list(open(input_file).read())
    vocabulary = sorted(list(set(raw_input)))
    encoder = tf.keras.layers.StringLookup(vocabulary=vocabulary)
    enc_input = encoder(np.array(raw_input)).numpy()
    return enc_input, encoder

def create_training_model(vocabulary_size: int):
    """
    Returns a Keras generative transformer model ready for training.
    
    The model will accept as input arrays of integers that are ASCII characters that have been encoded by a Keras
    StringLookup layer. The input shape should be (batch_size, index/time).
    It will pass the input into a token and position embedding, then a series of Transformer layers, then
    a final layer normalization, and finally through a Dense layer that produces logits of shape
    (batch_size, index/time, vocabulary_size) that predict the next character in a text series.
    The model uses sparse categorical crossentropy as a loss function.
    """
    values_input = tf.keras.layers.Input(shape=(TIMESERIES_CONTEXT,))
    layer = TokenAndPositionEmbedding(vocabulary_size, TIMESERIES_CONTEXT, NUM_HEADS*HEAD_SIZE)(values_input)
    for _ in range(NUM_LAYERS):
        layer = TransformerSelfAttention(NUM_HEADS, HEAD_SIZE, DROPOUT)(layer)
    layer = tf.keras.layers.LayerNormalization()(layer)
    layer = tf.keras.layers.Dense(vocabulary_size)(layer)

    model = tf.keras.Model(inputs=values_input, outputs=layer)
    optimizer = tf.keras.optimizers.Adam(LEARNING_RATE)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=optimizer, loss=loss)
    return model


class ChooseCharacterLayer(tf.keras.layers.Layer):
    """
    This is a layer meant to be added onto a model after training to make it more user-friendly in production.
    It takes logits of shape (batch_size, index/time, vocabulary_size) and for each row in the batch, predicts
    one character that could be next in a sequence based on the log likelihoods. The output shape will be
    (batch_size, 1). You can initialize it with a set of valid characters, in which case it will only predict
    characters from that set.
    """
    def __init__(self, encoder: tf.keras.layers.StringLookup, valid_chars: str=None):
        """
        Initialize with a StringLookup encoder that's already been fit to data.
        Optionally, only allow certain valid characters to be included in the predictions.
        If valid_chars is None, then this layer will predict any of the possible characters.
        """
        super().__init__()
        self.encoder = encoder
        if valid_chars is None:
            valid_chars = encoder.get_vocabulary()
        self.valid_chars = tf.constant(list(valid_chars))
        # Encode the valid characters so we can look up a subset of valid logits later on
        self._encoded_valid_chars = encoder(self.valid_chars).numpy()
    
    def call(self, logits: tf.Tensor):
        """
        Given a tensor of shape (batch_size, index/time, vocabulary_size), predict a next character for each
        row in the batch.
        """
        # Get a subset of the logits - every batch row but just the last character in the time sequence
        # and just the logits representing self.valid_chars
        logits_subset = tf.gather(logits[:,-1,:], self._encoded_valid_chars, axis=1)
        # Logits shape=(B,T,V) (batch, time, vocabulary)
        # For each batch, pick an item from the last time column, based on the probability given by logits in the V dimension
        choose_tokens = tf.random.categorical(logits_subset, num_samples=1)  # shape=(B,1)
        # Then do a reverse lookup of the indices to get characters and reshape so it's just one dimension
        return tf.gather(self.valid_chars, choose_tokens)


class LookupLayer(tf.keras.layers.Layer):
    """
    Encode input ASCII characters to integers.
    """
    def __init__(self, encoder):
        super().__init__()
        init = tf.lookup.KeyValueTensorInitializer(
            keys=tf.constant(encoder.get_vocabulary(), dtype=tf.string, name='input_keys'),
            values=tf.constant(range(92), dtype=tf.int64, name='input_vals')
        )
        self.table = tf.lookup.StaticVocabularyTable(init, num_oov_buckets=1)
    
    def call(self, values):
        return self.table[values]


def create_production_model(model, encoder, valid_chars=None):
    """
    Create a production-ready model given a model used for training.
    The difference is the production-ready model has a StringLookup encoder and decoder
    on the input and output, so you can pass it characters in and it will give you characters back
    """
    # Create a new input layer that accepts strings
    # Run that into the encoder
    # Strip the input layer off the training model and stick that in the middle
    # Add a decoding layer at the end that chooses characters
    # prod_model = tf.keras.Sequential([tf.keras.layers.Input(shape=(TIMESERIES_CONTEXT,), dtype=tf.string),
    #                                   encoder,
    #                                   *model.layers[1:],
    #                                   ChooseCharacterLayer(encoder, valid_chars)
    #                                  ])

    # prod_model = tf.keras.Sequential([tf.keras.layers.Input(shape=(TIMESERIES_CONTEXT,), name='values_input', dtype=tf.string),
    #                                   LookupLayer(encoder),
    #                                   *model.layers[1:],
    #                                   ChooseCharacterLayer(encoder, valid_chars)
    #                                  ])
    prod_model = tf.keras.Sequential([tf.keras.layers.Input(shape=(TIMESERIES_CONTEXT,), name='values_input', dtype=tf.string),
                                      encoder,
                                      *model.layers[1:],
                                      ChooseCharacterLayer(encoder, valid_chars)
                                     ])
    return prod_model

def generate_text(prod_model: tf.keras.Model, context: str, num_chars: int, print_output=True, stop_char=None) -> str:
    """
    Generates text using a trained model
    @param[in] prod_model: Model that takes characters as input and produces characters as output
                           (the model's prediction of the next character)
    @param[in] context: string to pass to the model the model; the model will generate its prediction of next character
    @param[in] num_chars: number of characters to generate
    @param[in] stop_char: stop when the model generates this character
    """
    if print_output:
        print(context, end='')
    for _ in range(num_chars):
        # Pad the context so it's at least the right length
        tokens = ['[UNK]']*(max(0, TIMESERIES_CONTEXT - len(context))) + list(context)
        tokens = tokens[-TIMESERIES_CONTEXT:]  # truncate
        char = prod_model.predict([tokens], verbose=0)[0,0].decode()
        if print_output:
            print(char, end='', flush=True)
        context += char
        if stop_char is not None and char == stop_char:
            break
    if print_output:
        print('')
    return ''.join(context)


def convert_to_tflite(model):
    """
    Convert a TF model to a quantized TF Lite model
    This reduces the size and will make it run faster at inference time by
    quantizing the internal weights dynamically

    Note: the conversion options turn on select TF operations during the conversion, meaning whatever inference
    platform this is running on needs to have a tflite runtime with full tensorflow operations enabled.
    """
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # This is adding select TensorFlow operations so the TF runtime needs to include
    # TF operations (https://www.tensorflow.org/lite/guide/ops_select)
    # This isn't an issue if we use the Python tf.lite module installed with the tensorflow pip package
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
                                           tf.lite.OpsSet.SELECT_TF_OPS]
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    return converter.convert()


def save_model_as_tflite(keras_model: tf.keras.Model, filename: str):
    """
    Convert a Keras model to a TF Lite model and save it to a file.
    """
    tflite_model = convert_to_tflite(keras_model)
    with open(filename, 'wb') as out_file:
        out_file.write(tflite_model)


def main(args=None):
    if args is None:
        parser = argparse.ArgumentParser(__file__.__doc__)
        parser.add_argument('--input_file', '-i',
                            help='Input file to train on (default: %s)' % INPUT_FILE,
                            default=INPUT_FILE)
        parser.add_argument('--model_file', '-o', required=True, help='Filename for saving the trained model')
        parser.add_argument('--training_steps', '-s', type=int,
                            help='Training steps/samples to use (default: all of the batches in the input file)')
        args = parser.parse_args()

    enc_input, encoder = read_input(args.input_file)
    model = create_training_model(encoder.vocabulary_size())
    model.summary()
    model.fit(generate_batch(enc_input, TIMESERIES_CONTEXT, BATCH_SIZE), steps_per_epoch=args.training_steps)

    # Create a production-ready model with encoder and decoder at the front and back
    prod_model = create_production_model(model, encoder, valid_chars=string.ascii_uppercase + '.')
    save_model_as_tflite(prod_model, args.model_file)


if __name__ == '__main__':
    main()

