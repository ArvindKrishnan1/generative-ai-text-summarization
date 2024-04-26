from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Dense, Dropout, GlobalAveragePooling1D, LayerNormalization, MultiHeadAttention
from tensorflow.keras.models import Model
from transformers import DistilBertTokenizer

# Initialize tokenizer for custom model
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

# Parameters
max_len_input = 100
vocab_size = 10000

# Positional Encoding
"""This function generates positional encodings for sequences.
Positional encodings are added to the input embeddings to provide information
about the position of tokens in the sequence. It implements the formula for
calculating sinusoidal positional encodings."""
def get_positional_encoding(seq_length, d_model):
    positions = tf.range(seq_length, dtype=tf.float32)[:, tf.newaxis]
    i = tf.range(d_model, dtype=tf.float32)[tf.newaxis, :]
    angle_rates = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
    angle_rads = positions * angle_rates
    sines = tf.math.sin(angle_rads[:, 0::2])
    cosines = tf.math.cos(angle_rads[:, 1::2])
    pos_encoding = tf.concat([sines, cosines], axis=-1)
    pos_encoding = pos_encoding[tf.newaxis, ...]
    return pos_encoding

#Transformer Block
"""This function defines a single transformer block, which consists of
multi-head self-attention mechanism followed by a
feed-forward neural network (FFNN). It applies layer normalization and dropout
for regularization. The multi-head self-attention mechanism allows the model to
 attend to different parts of the input sequence simultaneously, capturing
 dependencies between tokens.
"""
def transformer_block(x, num_heads, d_model, dff, rate, training):
    attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(x, x)
    attn_output = Dropout(rate)(attn_output, training=training)
    out1 = LayerNormalization(epsilon=1e-6)(x + attn_output)
    ffn_output = Dense(dff, activation='relu')(out1)
    ffn_output = Dense(d_model)(ffn_output)
    ffn_output = Dropout(rate)(ffn_output, training=training)
    return LayerNormalization(epsilon=1e-6)(out1 + ffn_output)


# Build the Transformer Model for Text Classification
"""This function constructs the Transformer model for text classification.
It creates the input layer, embedding layer, positional encoding, multiple
transformer blocks, global average pooling layer, and output layer. The model
is compiled with binary cross-entropy loss and Adam optimizer.
"""
def build_model(max_len_input, vocab_size, num_heads=8, d_model=128, dff=512, rate=0.1):
    # Input
    inputs = Input(shape=(max_len_input,), name="input")
    embedding = Embedding(vocab_size, d_model, name="embedding")(inputs)
    pos_encoding = get_positional_encoding(max_len_input, d_model)
    embedding += pos_encoding

    # Transformer Encoder
    encoder_output = embedding
    for _ in range(4):
        encoder_output = transformer_block(encoder_output, num_heads, d_model, dff, rate, training=True)

    # Global Average Pooling
    pooled_output = GlobalAveragePooling1D()(encoder_output)

    # Output layer
    outputs = Dense(1, activation="sigmoid")(pooled_output)  # Binary classification, use sigmoid activation

    # Define the model
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])  # Binary classification, use binary_crossentropy

    return model

# Create the model
model = build_model(max_len_input, vocab_size)
model.summary()

#Preprocess Data
"""This function preprocesses input text by tokenizing and encoding it using
the provided tokenizer. It ensures that the input sequence is padded or
truncated to the specified maximum length.
"""
def preprocess_data(input_text, tokenizer, max_len_input):
    # Tokenize and encode input text
    input_ids = tokenizer.encode(input_text, max_length=max_len_input, truncation=True)
    input_ids_padded = input_ids + [0] * (max_len_input - len(input_ids))  # Pad sequences
    return input_ids_padded

#Predict Class
"""This function predicts the class label for a given input text using the
trained Transformer model. It preprocesses the input text, converts it into a
tensor, and then passes it through the model to obtain the
predicted class label."""
def predict_class(input_text, tokenizer, model, max_len_input):
    input_ids_padded = preprocess_data(input_text, tokenizer, max_len_input)
    # Convert input to tensor
    input_ids_tensor = tf.convert_to_tensor([input_ids_padded])
    # Predict using the model
    outputs = model(input_ids_tensor)
    predicted_class = tf.argmax(outputs[0]).numpy()
    return predicted_class

predicted_class = predict_class("The US is in running out of oil", tokenizer, model, max_len_input)
print(predicted_class)
"""We began by crafting a Transformer-based architecture in PyTorch,
specifically tailored for text classification. However, due to time
constraints and our greater familiarity with Keras TensorFlow, we
pivoted to implementing the architecture using Keras TensorFlow. The outlined
steps demonstrate our approach to building the custom Transformer model for
text classification.
"""