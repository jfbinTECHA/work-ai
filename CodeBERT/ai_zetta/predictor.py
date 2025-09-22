# Predictor script for code completion using CodeBERT
# This script loads a (pre-trained or fine-tuned) model and tokenizer, and generates code completions for partial snippets.

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load pre-trained model and tokenizer
model = AutoModelForCausalLM.from_pretrained('microsoft/codebert-base')
tokenizer = AutoTokenizer.from_pretrained('microsoft/codebert-base')

# Function to predict code completions
def predict(code):
	# Encode the input code snippet
	encodings = tokenizer.encode_plus(
		code,
		return_tensors='pt'
	)

	# Generate predicted tokens (code completion)
	output = model.generate(**encodings, max_new_tokens=100)

	# Decode the generated tokens into a string
	completion = tokenizer.decode(output[0], skip_special_tokens=True)

	return completion
