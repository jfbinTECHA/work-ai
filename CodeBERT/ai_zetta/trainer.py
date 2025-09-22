# Trainer script for fine-tuning CodeBERT on code completion/classification tasks
# This script demonstrates loading data, splitting, training, validating, and saving a model

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. Load dataset of labeled examples (placeholder)
# Replace this with actual data loading logic
# Example: dataset = pd.read_csv('data.csv')
dataset = {
	'text': [
		'def add(a, b): return a + b',
		'def sub(a, b): return a - b',
		'def mul(a, b): return a * b',
		'def div(a, b): return a / b',
	],
	'labels': [0, 1, 2, 3]  # Example labels for demonstration
}

# 2. Split dataset into training and validation sets
train_texts, val_texts, train_labels, val_labels = train_test_split(
	dataset['text'], dataset['labels'], test_size=0.2, random_state=42
)

# 3. Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('microsoft/codebert-base')
model = AutoModelForSequenceClassification.from_pretrained('microsoft/codebert-base', num_labels=4)

# 4. Convert data into format expected by model
train_encodings = tokenizer(train_texts, truncation=True, padding=True, return_tensors='pt')
val_encodings = tokenizer(val_texts, truncation=True, padding=True, return_tensors='pt')
train_labels = torch.tensor(train_labels)
val_labels = torch.tensor(val_labels)

# 5. Train the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

model.train()
for epoch in range(3):  # Number of epochs
	optimizer.zero_grad()
	input_ids = train_encodings['input_ids'].to(device)
	attention_mask = train_encodings['attention_mask'].to(device)
	labels = train_labels.to(device)
	outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
	loss = outputs.loss
	loss.backward()
	optimizer.step()
	print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# 6. Validate the trained model
model.eval()
with torch.no_grad():
	input_ids = val_encodings['input_ids'].to(device)
	attention_mask = val_encodings['attention_mask'].to(device)
	labels = val_labels.to(device)
	outputs = model(input_ids, attention_mask=attention_mask)
	predictions = torch.argmax(outputs.logits, dim=1)
	acc = accuracy_score(labels.cpu(), predictions.cpu())
	print(f"Validation Accuracy: {acc:.4f}")

# 7. Save the trained model for future use
model.save_pretrained('trained_codebert_model')
tokenizer.save_pretrained('trained_codebert_model')
