{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "35b1a282-38d9-46ed-9118-0806aa2b016e",
   "metadata": {},
   "source": [
    "# Setting Up DistilBert Pretrained Model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "2173d635-15ce-4937-8d4c-528045f55a7d",
   "metadata": {},
   "outputs": [],
   "source": [
    "from transformers import TrainingArguments\n",
    "from keras.losses import SparseCategoricalCrossentropy\n",
    "import tftrainer\n",
    "import torch\n",
    "from transformers import DistilBertForSequenceClassification, DistilBertTokenizer, DataCollatorWithPadding\n",
    "\n",
    "# Initialize pretrained model and tokenizer\n",
    "tokenizer = DistilBertTokenizer.from_pretrained(\"distilbert-base-uncased\") # We will distil-bert-uncased\n",
    "model = DistilBertForSequenceClassification.from_pretrained(\"distilbert-base-uncased\",\n",
    "                                                            num_labels=4, # There are 4 categories in the AG News dataset\n",
    "                                                            problem_type=\"multi_label_classification\" # Enable multi label classification problem type\n",
    "                                                           ).to(device) # Use GPU Acceleration if applicable\n",
    "\n",
    "# Example use for pretrained model and tokenizer\n",
    "inputs = tokenizer(\"Hello, my dog is cute\", # input text\n",
    "                   padding='longest', # padding the text to maximum input length\n",
    "                   return_tensors=\"pt\") # returns input as a pytorch tensor\n",
    "\n",
    "with torch.no_grad(): # for inferences, we disable gradients\n",
    "    logits = model(**inputs).logits  # unpacks pytorch tensor, the model creates an inference, and we store the raw (unnormalized) data in logits\n",
    "\n",
    "# Store predicted class IDs\n",
    "predicted_class_ids = torch.arange(0,logits.shape[-1])[torch.sigmoid(logits).squeeze(dim=0) > 0.5] # Generatesclass ids and extracts any that are above 0.5\n",
    "\n",
    "labels = torch.sum(\n",
    "                  torch.nn.functional.one_hot(predicted_class_ids[None, :].clone(), # add dimension to and clones predicted_class_ids\n",
    "                                              num_classes=4), # specify number of possible classes for vector size\n",
    "                  dim=1 # Sum elements across column 1 after one-hot encoding\n",
    "                  ).to(torch.float) # necesarry for compatibility with loss function\n",
    "\n",
    "loss = model(**inputs, labels=labels).loss # performs cross-entropy loss calculation\n",
    "\n",
    "# Print labels for input\n",
    "labels"
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
