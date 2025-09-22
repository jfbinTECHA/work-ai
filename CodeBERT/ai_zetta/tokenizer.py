
# Hugging Face tokenizer integration for CodeBERT
# This script demonstrates how to use a pre-trained tokenizer from Hugging Face's transformers library
# to encode and decode source code for downstream tasks.

from transformers import AutoTokenizer
import os

# Load the pre-trained CodeBERT tokenizer from Hugging Face
# This tokenizer is specifically trained for source code and natural language pairs
tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")

# Function to encode and decode source code
# Args:
#   code (str): The source code string to be tokenized
# Returns:
#   input_ids (np.ndarray): Encoded token IDs (padded/truncated to max_length)
#   attention_mask (np.ndarray): Attention mask indicating real tokens vs padding
#   decoded_code (str): The decoded string from the token IDs
def encode_decode(code):
	# Tokenize the input code string
	inputs = tokenizer.encode_plus(
		code,                      # The code to tokenize
		max_length=512,            # Maximum sequence length
		padding='max_length',      # Pad sequences to max_length
		truncation=True,           # Truncate sequences longer than max_length
		return_attention_mask=True,# Return attention mask for distinguishing padding
		return_tensors='pt'        # Return PyTorch tensors
	)
    
	# Convert PyTorch tensors to numpy arrays for easier manipulation
	input_ids = inputs['input_ids'].numpy()[0]
	attention_mask = inputs['attention_mask'].numpy()[0]
    
	# Decode the token IDs back to a string (for verification)
	decoded_code = tokenizer.decode(input_ids, skip_special_tokens=True)
    
	return input_ids, attention_mask, decoded_code
