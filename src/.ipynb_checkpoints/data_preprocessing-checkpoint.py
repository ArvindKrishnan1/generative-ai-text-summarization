{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "f6fb95d1-f84e-4b81-8caf-f4e5380e9e00",
   "metadata": {},
   "source": [
    "# Data Preprocessing"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "cad97b39-f820-4ca5-9ed6-36c0987712fe",
   "metadata": {},
   "outputs": [],
   "source": [
    "from transformers import TrainingArguments\n",
    "from keras.losses import SparseCategoricalCrossentropy\n",
    "import tftrainer\n",
    "import torch\n",
    "from transformers import DistilBertForSequenceClassification, DistilBertTokenizer, DataCollatorWithPadding\n",
    "from torch import cuda\n",
    "from datasets import Dataset\n",
    "\n",
    "# Remove title length from df_val and df_train\n",
    "# Only use if the Winsorizer was utilized in Feature Analysis\n",
    "\n",
    "# df_val = df_val.drop(['Title Length'], axis=1)\n",
    "# df_train = df_train.drop(['Title Length'], axis=1)\n",
    "\n",
    "# Rename columns\n",
    "df_train.rename(columns = {'Title':'text','Class Index':'labels'}, inplace = True)\n",
    "df_val.rename(columns = {'Title':'text','Class Index':'labels'}, inplace = True)\n",
    "\n",
    "# Convert df_train and df_val to a HuggingFace dataset for easier tokenization\n",
    "hugging_train = Dataset.from_pandas(df_train)\n",
    "hugging_val = Dataset.from_pandas(df_val)\n",
    "\n",
    "# Confirm conversion\n",
    "hugging_train\n",
    "hugging_val\n",
    "\n",
    "# Function that automatically tokenizes datasets\n",
    "def preprocess_function(examples):\n",
    "    \"\"\"\n",
    "    Tokenize the text to create input and attention data\n",
    "    \n",
    "    in -> dataset (columns = text, label)\n",
    "    out -> tokenized dataset (columns = text, label, input, attention)\n",
    "    \"\"\"\n",
    "    return tokenizer(examples[\"text\"], truncation=True)\n",
    "\n",
    "# Tokenize our datasets\n",
    "tokenized_train = hugging_train.map(preprocess_function, batched=True)\n",
    "tokenized_val = hugging_val.map(preprocess_function, batched=True)\n",
    "\n",
    "# Confirm shapes of datasets\n",
    "tokenized_train.shape\n",
    "tokenized_val.shape"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "venv",
   "language": "python",
   "name": "venv"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
