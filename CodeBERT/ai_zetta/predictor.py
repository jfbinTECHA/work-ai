# Enhanced Predictor script for code completion using CodeBERT and T5
# This script loads pre-trained models and tokenizers, and generates code completions for partial snippets.
# Features:
# - CodeBERT for code-specific generation
# - T5 for advanced text preprocessing and understanding
# - Fallback mechanisms for robust operation
# - Comprehensive logging and error handling

from transformers import AutoModelForCausalLM, AutoTokenizer, T5Tokenizer, T5ForConditionalGeneration
import torch
import logging
import argparse

# Define commandline arguments
parser = argparse.ArgumentParser()
parser.add_argument('-l', '--loglevel', help='Set the logging level',
                    choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], default='INFO')
args = parser.parse_args()

# Configure logging based on chosen log level
logging.basicConfig(level=getattr(logging, args.loglevel), format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load pre-trained models and tokenizers
logger.info("Loading pre-trained CodeBERT and T5 models...")

# Load CodeBERT for code generation
logger.debug("Loading CodeBERT model and tokenizer...")
codebert_model = AutoModelForCausalLM.from_pretrained('microsoft/codebert-base')
codebert_tokenizer = AutoTokenizer.from_pretrained('microsoft/codebert-base')
logger.debug("CodeBERT loaded successfully")

# Load T5 for enhanced text understanding
logger.debug("Loading T5 model and tokenizer...")
try:
    t5_model = T5ForConditionalGeneration.from_pretrained('t5-small')
    t5_tokenizer = T5Tokenizer.from_pretrained('t5-small')
    t5_available = True
    logger.debug("T5 loaded successfully")
except Exception as e:
    logger.warning(f"T5 model loading failed: {e}")
    logger.info("Falling back to CodeBERT-only predictions")
    t5_model = None
    t5_tokenizer = None
    t5_available = False

logger.info("Models loaded successfully")

class Predictor:
    """
    Enhanced code predictor using both CodeBERT and T5 models.

    This class provides intelligent code completion by leveraging:
    - T5 for advanced text preprocessing and understanding
    - CodeBERT for code-specific generation
    """

    def __init__(self, use_t5=True):
        """
        Initialize the predictor with specified models.

        Args:
            use_t5 (bool): Whether to use T5 preprocessing (default: True)
        """
        self.codebert_model = codebert_model
        self.codebert_tokenizer = codebert_tokenizer
        self.use_t5 = use_t5 and t5_available
        self.t5_model = t5_model
        self.t5_tokenizer = t5_tokenizer

        if self.use_t5:
            logger.info("T5 preprocessing enabled")
        else:
            logger.info("T5 preprocessing disabled - using CodeBERT only")

    def preprocess_with_t5(self, code_snippet):
        """
        Preprocess code using T5 for enhanced understanding.

        Args:
            code_snippet (str): Input code to preprocess

        Returns:
            str: T5-processed version of the code
        """
        if not self.use_t5 or self.t5_model is None:
            return code_snippet

        try:
            logger.debug("Preprocessing code with T5...")

            # Create T5 input format
            t5_input = f"translate code to natural language: {code_snippet}"

            # Encode with T5
            encoded_input = self.t5_tokenizer.encode_plus(
                t5_input,
                max_length=512,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )

            # Generate T5 output
            t5_outputs = self.t5_model.generate(
                encoded_input['input_ids'],
                attention_mask=encoded_input['attention_mask'],
                max_length=150,
                num_beams=4,
                early_stopping=True
            )

            # Decode T5 output
            natural_language = self.t5_tokenizer.decode(t5_outputs[0], skip_special_tokens=True)

            logger.debug(f"T5 preprocessing result: {natural_language}")

            # Create enhanced input for CodeBERT
            enhanced_input = f"{natural_language}\n{code_snippet}"

            return enhanced_input

        except Exception as e:
            logger.warning(f"T5 preprocessing failed: {e}")
            return code_snippet

    def predict(self, code_snippet, max_length=100):
        """
        Generate code completion using enhanced preprocessing.

        Args:
            code_snippet (str): Input code to complete
            max_length (int): Maximum length of generated code

        Returns:
            str: Generated code completion
        """
        logger.info(f"Generating completion for input: {code_snippet}")

        try:
            # Preprocess with T5 if available
            enhanced_input = self.preprocess_with_t5(code_snippet)
            logger.debug("Input preprocessing completed")

            # Encode with CodeBERT tokenizer
            logger.debug("Encoding with CodeBERT tokenizer...")
            encodings = self.codebert_tokenizer.encode_plus(
                enhanced_input,
                return_tensors='pt',
                max_length=512,
                truncation=True,
                padding='max_length'
            )

            # Generate with CodeBERT
            logger.debug("Generating code completion...")
            outputs = self.codebert_model.generate(
                **encodings,
                max_new_tokens=max_length,
                num_beams=4,
                early_stopping=True,
                do_sample=True,
                temperature=0.7
            )

            # Decode the generated tokens
            logger.debug("Decoding generated tokens...")
            completion = self.codebert_tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Clean up the completion (remove the original input if present)
            if enhanced_input in completion:
                completion = completion.replace(enhanced_input, "").strip()

            logger.info(f"Generated completion: {completion}")
            return completion

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            # Fallback to simple prediction without T5
            return self.simple_predict(code_snippet, max_length)

    def simple_predict(self, code_snippet, max_length=100):
        """
        Simple prediction without T5 preprocessing (fallback method).

        Args:
            code_snippet (str): Input code to complete
            max_length (int): Maximum length of generated code

        Returns:
            str: Generated code completion
        """
        logger.debug("Using simple prediction (no T5 preprocessing)")

        try:
            encodings = self.codebert_tokenizer.encode_plus(
                code_snippet,
                return_tensors='pt',
                max_length=512,
                truncation=True,
                padding='max_length'
            )

            outputs = self.codebert_model.generate(
                **encodings,
                max_new_tokens=max_length,
                num_beams=4,
                early_stopping=True
            )

            completion = self.codebert_tokenizer.decode(outputs[0], skip_special_tokens=True)
            return completion

        except Exception as e:
            logger.error(f"Simple prediction also failed: {e}")
            return f"# Error in prediction: {str(e)}"

# Legacy function for backward compatibility
def predict(code):
    """
    Legacy predict function for backward compatibility.

    Args:
        code (str): Input code snippet

    Returns:
        str: Generated completion
    """
    predictor = Predictor()
    return predictor.predict(code)

def init_predictor():
    """
    Initialize the enhanced code predictor component with T5 integration.

    This function performs initialization checks for both CodeBERT and T5 models,
    including verifying transformers availability and model loading capabilities.

    Returns:
        bool: True if initialization successful, False otherwise
    """
    try:
        logger.info("Initializing enhanced code predictor component...")

        # Check if required dependencies are available
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, T5Tokenizer, T5ForConditionalGeneration
            logger.debug("Transformers library available")
        except ImportError:
            logger.error("Transformers library not available - predictor functionality not available")
            return False

        try:
            import torch
            logger.debug("PyTorch available")
        except ImportError:
            logger.error("PyTorch not available - predictor functionality not available")
            return False

        # Check CUDA availability for model acceleration
        if torch.cuda.is_available():
            logger.info(f"CUDA available - model acceleration enabled ({torch.cuda.get_device_name()})")
        else:
            logger.info("CUDA not available - using CPU inference")

        # Try to load models (will be cached after first load)
        models_loaded = 0

        try:
            logger.debug("Loading CodeBERT model and tokenizer...")
            codebert_model = AutoModelForCausalLM.from_pretrained('microsoft/codebert-base')
            codebert_tokenizer = AutoTokenizer.from_pretrained('microsoft/codebert-base')
            models_loaded += 1
            logger.debug("CodeBERT loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load CodeBERT model: {e}")

        try:
            logger.debug("Loading T5 model and tokenizer...")
            t5_model = T5ForConditionalGeneration.from_pretrained('t5-small')
            t5_tokenizer = T5Tokenizer.from_pretrained('t5-small')
            models_loaded += 1
            logger.debug("T5 loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load T5 model: {e}")
            logger.info("T5 preprocessing will be disabled")

        if models_loaded > 0:
            logger.info(f"Code predictor initialized successfully ({models_loaded}/2 models loaded)")
            return True
        else:
            logger.error("No models could be loaded - predictor functionality limited")
            return False

    except Exception as e:
        logger.error(f"Failed to initialize code predictor: {e}")
        return False

# Example usage
if __name__ == "__main__":
    logger.info("Starting enhanced CodeBERT + T5 predictor...")

    # Initialize predictor with T5 preprocessing
    predictor = Predictor(use_t5=True)

    # Example code snippets for testing
    examples = [
        "def calculate_fibonacci(n):",
        "def sort_array(arr):",
        "class DataProcessor:",
        "def binary_search(items, target):"
    ]

    for i, example_code in enumerate(examples, 1):
        logger.info(f"\n--- Example {i} ---")
        logger.info(f"Input: {example_code}")

        try:
            completion = predictor.predict(example_code, max_length=150)
            logger.info(f"Generated: {completion}")

            # Show T5 preprocessing status
            if predictor.use_t5:
                logger.info("✓ Used T5 preprocessing for enhanced understanding")
            else:
                logger.info("○ Used CodeBERT-only prediction")

        except Exception as e:
            logger.error(f"Prediction failed for example {i}: {e}")

    logger.info("Enhanced prediction demonstration completed!")
