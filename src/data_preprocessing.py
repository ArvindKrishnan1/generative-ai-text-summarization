from transformers import TrainingArguments
from keras.losses import SparseCategoricalCrossentropy
import tftrainer
import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer, DataCollatorWithPadding
from torch import cuda
from datasets import Dataset

# Remove title length from df_val and df_train
# Only use if the Winsorizer was utilized in Feature Analysis
# df_val = df_val.drop(['Title Length'], axis=1)
# df_train = df_train.drop(['Title Length'], axis=1)

# Rename columns
df_train.rename(columns = {'Title':'text','Class Index':'labels'}, inplace = True)
df_val.rename(columns = {'Title':'text','Class Index':'labels'}, inplace = True)

# Convert df_train and df_val to a HuggingFace dataset for easier tokenization
hugging_train = Dataset.from_pandas(df_train)
hugging_val = Dataset.from_pandas(df_val)

# Confirm conversion
hugging_train
hugging_val

# Function that automatically tokenizes datasets
def preprocess_function(examples):
    """
    Tokenize the text to create input and attention data
    in -> dataset (columns = text, label)
    out -> tokenized dataset (columns = text, label, input, attention)
    \\\
    return tokenizer(examples[\text\], truncation=True)
    """
    # Tokenize our datasets
    tokenized_train = hugging_train.map(preprocess_function, batched=True)
    tokenized_val = hugging_val.map(preprocess_function, batched=True)
    
# Confirm shapes of datasets
tokenized_train.shape
tokenized_val.shape