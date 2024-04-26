from transformers import TrainingArguments
from keras.losses import SparseCategoricalCrossentropy
import tftrainer
import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer, DataCollatorWithPadding
    
# Initialize pretrained model and tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased') # We will distil-bert-uncased
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased',
                                                            num_labels=4, # There are 4 categories in the AG News dataset
                                                            problem_type='multi_label_classification' # Enable multi label classification problem type
                                                            ).to(device) # Use GPU Acceleration if applicable
    
# Example use for pretrained model and tokenizer
inputs = tokenizer("Hello, my dog is cute", # input text
                    padding='longest', # padding the text to maximum input length
                    return_tensors='pt') # returns input as a pytorch tensor
    
with torch.no_grad(): # for inferences, we disable gradients
        logits = model(**inputs).logits  # unpacks pytorch tensor, the model creates an inference, and we store the raw (unnormalized) data in logits
    
# Store predicted class IDs
predicted_class_ids = torch.arange(0,logits.shape[-1])[torch.sigmoid(logits).squeeze(dim=0) > 0.5] # Generatesclass ids and extracts any that are above 0.5
    
labels = torch.sum(
                    torch.nn.functional.one_hot(predicted_class_ids[None, :].clone(), # add dimension to and clones predicted_class_ids
                                                num_classes=4), # specify number of possible classes for vector size
                    dim=1 # Sum elements across column 1 after one-hot encoding
                    ).to(torch.float) # necesarry for compatibility with loss function
    
loss = model(**inputs, labels=labels).loss # performs cross-entropy loss calculation
    
# Print labels for input
labels