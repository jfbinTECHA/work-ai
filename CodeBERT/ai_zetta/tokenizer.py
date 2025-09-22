# Hugging Face tokenizer integration for CodeBERT
from transformers import AutoTokenizer
import os

# Load pre-trained tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")

# Define function for encoding and decoding source code
def encode_decode(code):
	inputs = tokenizer.encode_plus(
		code,
		max_length=512,
		padding='max_length',
		truncation=True,
		return_attention_mask=True,
		return_tensors='pt'
	)
    
	# Convert tensor to numpy array for easier manipulation
	input_ids = inputs['input_ids'].numpy()[0]
	attention_mask = inputs['attention_mask'].numpy()[0]
    
	decoded_code = tokenizer.decode(input_ids, skip_special_tokens=True)
    
	return input_ids, attention_mask, decoded_code
