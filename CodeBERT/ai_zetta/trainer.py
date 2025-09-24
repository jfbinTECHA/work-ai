# Trainer script for fine-tuning CodeBERT on code completion/classification tasks
# This script demonstrates loading data, splitting, training, validating, and saving a model

import torch
import logging
import argparse
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Define commandline arguments
parser = argparse.ArgumentParser()
parser.add_argument('-l', '--loglevel', help='Set the logging level',
                    choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], default='INFO')
args = parser.parse_args()

# Configure logging based on chosen log level
logging.basicConfig(level=getattr(logging, args.loglevel), format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 1. Load dataset of labeled examples (placeholder)
# Replace this with actual data loading logic
# Example: dataset = pd.read_csv('data.csv')
logger.info("Loading dataset...")
dataset = {
 	'text': [
 		'def add(a, b): return a + b',
 		'def sub(a, b): return a - b',
 		'def mul(a, b): return a * b',
 		'def div(a, b): return a / b',
 	],
 	'labels': [0, 1, 2, 3]  # Example labels for demonstration
}
logger.info(f"Loaded dataset with {len(dataset['text'])} examples")

# 2. Split dataset into training and validation sets
logger.info("Splitting dataset into training and validation sets...")
train_texts, val_texts, train_labels, val_labels = train_test_split(
 	dataset['text'], dataset['labels'], test_size=0.2, random_state=42
)
logger.info(f"Training set: {len(train_texts)} examples, Validation set: {len(val_texts)} examples")

# 3. Initialize tokenizer and model
logger.info("Initializing tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained('microsoft/codebert-base')
model = AutoModelForSequenceClassification.from_pretrained('microsoft/codebert-base', num_labels=4)
logger.info("Model and tokenizer initialized successfully")

# 4. Convert data into format expected by model
logger.info("Converting data to model format...")
train_encodings = tokenizer(train_texts, truncation=True, padding=True, return_tensors='pt')
val_encodings = tokenizer(val_texts, truncation=True, padding=True, return_tensors='pt')
train_labels = torch.tensor(train_labels)
val_labels = torch.tensor(val_labels)
logger.info("Data conversion completed")

# 5. Train the model
logger.info("Setting up training...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
logger.info("Starting training...")

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
	logger.info(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# 6. Validate the trained model
logger.info("Running validation...")
model.eval()
with torch.no_grad():
	input_ids = val_encodings['input_ids'].to(device)
	attention_mask = val_encodings['attention_mask'].to(device)
	labels = val_labels.to(device)
	outputs = model(input_ids, attention_mask=attention_mask)
	predictions = torch.argmax(outputs.logits, dim=1)
	acc = accuracy_score(labels.cpu(), predictions.cpu())
	logger.info(f"Validation Accuracy: {acc:.4f}")

# 7. Save the trained model for future use
logger.info("Saving trained model and tokenizer...")
model.save_pretrained('trained_codebert_model')
tokenizer.save_pretrained('trained_codebert_model')
logger.info("Model and tokenizer saved successfully to 'trained_codebert_model'")

def init_trainer():
    """
    Initialize the model trainer component.

    This function performs basic initialization checks for the trainer
    including verifying PyTorch and transformers availability.

    Returns:
        bool: True if initialization successful, False otherwise
    """
    try:
        logger.info("Initializing model trainer component...")

        # Check if required dependencies are available
        try:
            import torch
            logger.debug("PyTorch available")
        except ImportError:
            logger.error("PyTorch not available - trainer functionality not available")
            return False

        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            logger.debug("Transformers library available")
        except ImportError:
            logger.error("Transformers library not available - trainer functionality not available")
            return False

        try:
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import accuracy_score
            logger.debug("Scikit-learn available")
        except ImportError:
            logger.warning("Scikit-learn not available - some training features may be limited")

        # Check CUDA availability
        if torch.cuda.is_available():
            logger.info(f"CUDA available - GPU training supported ({torch.cuda.get_device_name()})")
        else:
            logger.info("CUDA not available - using CPU training")

        logger.info("Model trainer component initialized successfully")
        return True

    except Exception as e:
        logger.error(f"Failed to initialize model trainer: {e}")
        return False
