from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
import os
import re
import string
from typing import Dict, List, Any, Optional, Tuple
import logging
from dotenv import load_dotenv

import torch
import warnings
import numpy as np
import google.generativeai as genai
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)

# Text preprocessing imports
import nltk

# Don't download NLTK data at module level - do it later
NLTK_INITIALIZED = False

def initialize_nltk():
    """Initialize NLTK data with robust error handling"""
    global NLTK_INITIALIZED
    
    if NLTK_INITIALIZED:
        return True
    
    try:
        # Try to download required NLTK data
        required_downloads = ['punkt', 'punkt_tab', 'stopwords', 'wordnet', 'omw-1.4']
        
        for item in required_downloads:
            try:
                nltk.download(item, quiet=True)
            except Exception as e:
                print(f"Warning: Could not download {item}: {e}")
        
        NLTK_INITIALIZED = True
        print("✅ NLTK initialized successfully")
        return True
        
    except Exception as e:
        print(f"❌ NLTK initialization failed: {e}")
        print("📝 Continuing with basic preprocessing...")
        NLTK_INITIALIZED = False
        return False

# Importing Transformer Interpret for explainability
try:
    from transformers_interpret import SequenceClassificationExplainer
    TRANSFORMER_INTERPRET_AVAILABLE = True
except ImportError:
    TRANSFORMER_INTERPRET_AVAILABLE = False
    print("Warning: transformers_interpret not available. Install with: pip install transformers-interpret")

# Suppress warnings
warnings.filterwarnings("ignore")

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="Climate Misinformation Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    text: str

class DetectionResult(BaseModel):
    is_climate_related: bool
    is_complete_statement: Optional[bool] = None
    completeness_guidance: Optional[str] = None
    model_prediction: Optional[Dict[str, Any]] = None
    final_result: Optional[Dict[str, Any]] = None
    explanation: Optional[str] = None
    confidence: Optional[float] = None
    word_attributions: Optional[List[Dict[str, Any]]] = None
    status: str  # "complete_analysis", "incomplete_statement", "out_of_domain"
    # Debug info (optional)
    preprocessing_info: Optional[Dict[str, str]] = None

# Global variables
pipeline = None
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_PATH = os.getenv("MODEL_PATH")

class TextPreprocessor:
    """
    Text preprocessing pipeline matching training data preprocessing
    """
    def __init__(self, use_lemmatization=True):
        self.use_lemmatization = use_lemmatization
        self.nltk_available = False
        
        # Initialize NLTK components with fallbacks
        self._setup_nltk_components()
        
    def _setup_nltk_components(self):
        """Setup NLTK components with fallback options"""
        try:
            # Try to initialize NLTK
            initialize_nltk()
            
            # Try to import and setup NLTK components
            from nltk.corpus import stopwords
            from nltk.tokenize import word_tokenize
            from nltk.stem import WordNetLemmatizer
            
            # Initialize stopwords
            try:
                self.stop_words = set(stopwords.words('english'))
                self.nltk_available = True
            except Exception:
                # Fallback stopwords
                self.stop_words = self._get_fallback_stopwords()
                self.nltk_available = False
            
            # Remove important climate-related words from stopwords
            climate_important_words = {'not', 'no', 'never', 'without', 'against', 'false', 'fake', 'real', 'true'}
            self.stop_words = self.stop_words - climate_important_words
            
            # Initialize lemmatizer
            if self.use_lemmatization:
                try:
                    self.lemmatizer = WordNetLemmatizer()
                    # Test the lemmatizer
                    test_word = self.lemmatizer.lemmatize("testing")
                except Exception:
                    logger.warning("WordNet lemmatizer not available, disabling lemmatization")
                    self.use_lemmatization = False
                    
            logger.info(f"TextPreprocessor initialized (NLTK available: {self.nltk_available})")
            
        except Exception as e:
            logger.warning(f"NLTK setup failed: {e}. Using fallback preprocessing.")
            self.stop_words = self._get_fallback_stopwords()
            self.use_lemmatization = False
            self.nltk_available = False
    
    def _get_fallback_stopwords(self):
        """Fallback stopwords if NLTK not available"""
        return set([
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'should', 'could', 'can', 'may', 'might', 'must', 'this', 'that', 'these', 'those',
            'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his',
            'her', 'its', 'our', 'their', 'mine', 'yours', 'his', 'hers', 'ours', 'theirs'
        ])
    
    def _simple_tokenize(self, text):
        """Simple tokenization fallback"""
        # Remove punctuation and split
        import re
        text = re.sub(r'[^\w\s]', ' ', text)
        return text.split()
    
    def clean_text(self, text: str) -> str:
        """
        Comprehensive text cleaning pipeline
        """
        if not text or text.strip() == '':
            return ''
        
        # Convert to string if not already
        text = str(text)
        
        # 1. URL cleaning - remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        text = re.sub(r'www\.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # 2. Hashtag cleaning - remove # but keep the word
        text = re.sub(r'#(\w+)', r'\1', text)
        
        # 3. Mention cleaning - remove @mentions
        text = re.sub(r'@\w+', '', text)
        
        # 4. Remove extra whitespace and newlines
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n', ' ', text)
        
        # 5. Remove numbers (but keep important ones like CO2, etc.)
        # Keep numbers that are part of scientific terms
        text = re.sub(r'\b(?<!CO)\d+(?!°|%|C|F|ppm)\b', '', text)
        
        # 6. Convert to lowercase
        text = text.lower()
        
        # 7. Remove punctuation but preserve important climate-related symbols
        # Keep degree symbols, percentages, etc.
        translator = str.maketrans('', '', string.punctuation.replace('°', '').replace('%', ''))
        text = text.translate(translator)
        
        # 8. Remove extra spaces
        text = ' '.join(text.split())
        
        return text.strip()
    
    def tokenize_and_process(self, text: str) -> List[str]:
        """
        Tokenize text and apply stemming/lemmatization
        """
        if not text:
            return []
        
        # Tokenize with error handling
        try:
            if self.nltk_available:
                from nltk.tokenize import word_tokenize
                tokens = word_tokenize(text)
            else:
                tokens = self._simple_tokenize(text)
        except Exception as e:
            logger.debug(f"Tokenization error: {e}, using simple split")
            tokens = self._simple_tokenize(text)
        
        # Remove stopwords and short words (but keep negations)
        tokens = [token for token in tokens if token not in self.stop_words and len(token) > 2]
        
        # Apply lemmatization if available
        try:
            if self.use_lemmatization and hasattr(self, 'lemmatizer'):
                tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        except Exception as e:
            logger.debug(f"Lemmatization error: {e}")
            pass
        
        return tokens
    
    def preprocess_text(self, text: str) -> str:
        """
        Complete preprocessing pipeline matching training data
        """
        # Clean text
        cleaned_text = self.clean_text(text)
        
        # Tokenize and process
        tokens = self.tokenize_and_process(cleaned_text)
        
        # Join back to string
        processed_text = ' '.join(tokens)
        
        return processed_text

class ClimateDetectionPipeline:
    def __init__(self):
        self.gemini_model = None
        self.trained_model = None
        self.tokenizer = None
        self.transformer_explainer = None
        
        # Initialize text preprocessor
        self.text_preprocessor = TextPreprocessor(use_lemmatization=True)
        
        logger.info("Initializing Climate Detection Pipeline...")
        
        self.setup_gemini()
        self.load_trained_model()
        self.setup_xai()
        
        logger.info("Pipeline initialization complete!")

    def setup_gemini(self):
        """Initialize Gemini API with correct model name"""
        try:
            logger.info("Setting up Gemini...")
            
            if not GEMINI_API_KEY or GEMINI_API_KEY == "your-gemini-api-key-here":
                logger.warning("GEMINI_API_KEY not set")
                return
            
            genai.configure(api_key=GEMINI_API_KEY)
            
            model_names = [
                'gemini-2.5-pro'
            ]
            
            for model_name in model_names:
                try:
                    self.gemini_model = genai.GenerativeModel(model_name)
                    # Test with a simple prompt
                    test_response = self.gemini_model.generate_content("Say 'test'")
                    logger.info(f"Gemini connected successfully with model: {model_name}")
                    break
                except Exception as e:
                    logger.debug(f"Model {model_name} failed: {e}")
                    continue
            
            if not self.gemini_model:
                # List available models for debugging
                try:
                    models = genai.list_models()
                    available_models = [m.name for m in models if 'generateContent' in m.supported_generation_methods]
                    logger.error(f"No working Gemini model found. Available models: {available_models[:3]}")
                except:
                    logger.error("Could not list available Gemini models")
            
        except ImportError:
            logger.error("google-generativeai package not installed")
        except Exception as e:
            logger.error(f"Gemini setup failed: {e}")

    def load_trained_model(self):
        
        try:
            logger.info("Loading trained ClimateBERT model...")

            if not os.path.exists(MODEL_PATH):
                logger.error(f"Model path not found: {MODEL_PATH}")
                raise FileNotFoundError(f"Model path not found: {MODEL_PATH}")

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                # Load tokenizer and model directly
                logger.info(f"Loading model from: {MODEL_PATH}")
                self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
                self.trained_model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
                self.trained_model.eval()
                
                logger.info("✅ Model and tokenizer loaded successfully")

            # Test the model and check output shape
            test_input = self.tokenizer("test climate change", return_tensors="pt", truncation=True, padding=True)
            with torch.no_grad():
                outputs = self.trained_model(**test_input)
                logits = outputs.logits
                probs = torch.nn.functional.softmax(logits, dim=-1)

                logger.info(f"✅ Model loaded - Output shape: {probs.shape}")
                logger.info(f"✅ Number of classes: {probs.shape[-1]}")

                # Verify we have exactly 2 classes for binary classification
                if probs.shape[-1] != 2:
                    raise ValueError(f"Expected 2 classes, but model outputs {probs.shape[-1]} classes")
                
                logger.info("✅ Model verification complete - Ready for inference")

        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            raise e  # Re-raise the exception to stop execution

    def setup_xai(self):
        """Setup XAI components with Transformer Interpret"""
        try:
            if not self.trained_model:
                logger.warning("Skipping XAI setup - model not loaded")
                return
            
            logger.info("Setting up XAI components...")
            
            # Setup Transformer Interpret
            if TRANSFORMER_INTERPRET_AVAILABLE:
                try:
                    self.transformer_explainer = SequenceClassificationExplainer(
                        self.trained_model, 
                        self.tokenizer
                    )
                    logger.info("Transformer Interpret initialized successfully")
                except Exception as e:
                    logger.error(f"Transformer Interpret setup failed: {e}")
                    self.transformer_explainer = None
            else:
                logger.warning("Transformer Interpret not available")
                self.transformer_explainer = None
            
        except Exception as e:
            logger.error(f"XAI setup failed: {e}")

    def check_climate_domain_and_completeness(self, original_text: str) -> Dict[str, Any]:
        """
        Single Gemini call for both domain check and completeness evaluation
        Uses ORIGINAL text (not preprocessed)
        """
        try:
            if not self.gemini_model:
                # Keyword-based fallback for domain check
                logger.info("Gemini model not available, using keyword-based domain check")
                climate_keywords = [
                    'climate', 'warming', 'carbon', 'temperature', 'greenhouse', 
                    'emissions', 'co2', 'fossil', 'renewable', 'environment',
                    'pollution', 'weather', 'arctic', 'ice', 'ocean', 'atmosphere',
                    'sea level', 'glacier', 'wildfire', 'drought', 'flood'
                ]
                text_lower = original_text.lower()
                is_climate = any(keyword in text_lower for keyword in climate_keywords)
                
                return {
                    "is_climate_related": is_climate,
                    "is_complete": True if is_climate else None,
                    "guidance": None,
                    "reasoning": "Keyword-based fallback - Gemini unavailable"
                }
            
            # Combined domain and completeness check prompt (using original text)
            prompt = f"""You are evaluating a statement for climate misinformation detection. Perform a two-stage analysis:

**STAGE 1: CLIMATE DOMAIN CHECK**
Determine if this statement relates to climate change or environmental science, including:
- Global warming, greenhouse gases, CO₂ emissions, temperature changes
- Sea level rise, melting ice, extreme weather, fossil fuels
- Renewable energy, deforestation, ocean acidification, carbon footprint
- Other environmental changes linked to Earth's climate system

**STAGE 2: COMPLETENESS CHECK (only if climate-related)**
If the statement IS climate-related, evaluate if it contains enough information for scientific evidence evaluation.

A COMPLETE statement must be:
1. **Scientifically verifiable** - Can be checked against peer-reviewed research, official climate reports (IPCC, NASA, NOAA)
2. **Factually grounded** - Makes claims about measurable phenomena, scientific processes, or documented events
3. **Source-independent** - Does not rely on unverifiable videos, anecdotal stories, or non-scientific claims
4. **Specific enough** - Contains sufficient detail for scientific evaluation
5. **No vague references** - Does not use undefined pronouns or references like "his results", "their study", "this research" without clear identification

An INCOMPLETE statement:
- References unverifiable sources (random videos, anecdotal stories, rumors)
- Makes claims that cannot be checked against scientific literature
- Relies on subjective opinions or non-scientific evidence
- Lacks specificity needed for scientific verification
- **Contains vague pronoun references** (his/her/their/this/that research/results/study) without identifying the specific person, institution, or study
- Uses undefined references like "scientists say", "studies show", "research proves" without naming specific sources

**NOW ANALYZE THIS STATEMENT:**

Statement: "{original_text}"

**REQUIRED RESPONSE FORMAT:**
DOMAIN: [CLIMATE_RELATED or NOT_CLIMATE_RELATED]
STATUS: [COMPLETE or INCOMPLETE] (only if DOMAIN is CLIMATE_RELATED)
REASONING: [Detailed explanation focusing on scientific verifiability] (only if DOMAIN is CLIMATE_RELATED)
GUIDANCE: [If incomplete, suggest how to make it scientifically verifiable. If complete, write "None"] (only if DOMAIN is CLIMATE_RELATED)"""

            response = self.gemini_model.generate_content(prompt)
            response_text = response.text.strip()
            logger.info(f"Combined domain/completeness response: {response_text}")
            
            # Parse the response
            lines = response_text.split('\n')
            domain_line = next((line for line in lines if line.startswith('DOMAIN:')), '')
            status_line = next((line for line in lines if line.startswith('STATUS:')), '')
            reasoning_line = next((line for line in lines if line.startswith('REASONING:')), '')
            guidance_line = next((line for line in lines if line.startswith('GUIDANCE:')), '')
            
            # Extract domain result
            is_climate_related = 'CLIMATE_RELATED' in domain_line.upper()
            
            if not is_climate_related:
                logger.info("Statement not climate-related")
                return {
                    "is_climate_related": False,
                    "is_complete": None,
                    "guidance": None,
                    "reasoning": "Statement is not related to climate change or environmental science"
                }
            
            # Extract completeness result (only for climate-related statements)
            is_complete = 'COMPLETE' in status_line.upper() and 'INCOMPLETE' not in status_line.upper()
            reasoning = reasoning_line.replace('REASONING:', '').strip() if reasoning_line else "Analysis completed"
            guidance = guidance_line.replace('GUIDANCE:', '').strip() if guidance_line else None
            
            if guidance and guidance.lower() in ['none', 'n/a', '']:
                guidance = None
            
            logger.info(f"Climate-related statement: {'COMPLETE' if is_complete else 'INCOMPLETE'}")
            if not is_complete:
                logger.info(f"Guidance: {guidance}")
            
            return {
                "is_climate_related": True,
                "is_complete": is_complete,
                "reasoning": reasoning,
                "guidance": guidance
            }
            
        except Exception as e:
            logger.error(f"Combined domain/completeness check error: {e}")
            return {
                "is_climate_related": True,  # Default to allowing analysis
                "is_complete": True,
                "reasoning": "Error in combined check - proceeding with analysis",
                "guidance": None
            }

    def get_model_prediction(self, preprocessed_text: str) -> Dict[str, Any]:
        """
        Get binary model prediction (SUPPORT/REFUTE)
        Uses PREPROCESSED text for consistent training/inference
        """
        try:
            if not self.trained_model or not self.tokenizer:
                return {
                    "SUPPORT": 0.5,
                    "REFUTE": 0.5,
                    "reasoning": "Model unavailable"
                }
            
            logger.info(f"🔧 Model input (preprocessed): '{preprocessed_text}'")
            
            inputs = self.tokenizer(
                preprocessed_text, 
                return_tensors="pt", 
                truncation=True, 
                padding=True,
                max_length=512
            )
            
            with torch.no_grad():
                outputs = self.trained_model(**inputs)
                logits = outputs.logits
                probs = torch.nn.functional.softmax(logits, dim=-1)
                
                # Handle different output shapes
                if probs.dim() > 1:
                    probs = probs.squeeze()
                
                probs_np = probs.cpu().numpy()
                
                # Ensure we have a 1D array
                if probs_np.ndim > 1:
                    probs_np = probs_np[0]
                
                logger.debug(f"Model output shape: {probs_np.shape}, values: {probs_np}")
                
                # Handle different numbers of output classes
                if len(probs_np) == 2:
                    # Perfect - binary classification
                    result = {
                        "SUPPORT": float(probs_np[1]),  # Assuming index 0 is SUPPORT
                        "REFUTE": float(probs_np[0])   # Assuming index 1 is REFUTE
                    }
                    
                elif len(probs_np) == 3:
                    # Legacy 3-class model - combine NOT_ENOUGH_INFO with lower confidence class
                    support_score = float(probs_np[1])
                    refute_score = float(probs_np[0])
                    neutral_score = float(probs_np[2])
                    
                    # Distribute neutral score proportionally
                    if support_score >= refute_score:
                        support_score += neutral_score * 0.3
                        refute_score += neutral_score * 0.7
                    else:
                        support_score += neutral_score * 0.7
                        refute_score += neutral_score * 0.3
                    
                    result = {
                        "SUPPORT": support_score,
                        "REFUTE": refute_score
                    }
                else:
                    logger.warning(f"Unexpected model output size: {len(probs_np)}")
                    return {
                        "SUPPORT": 0.5,
                        "REFUTE": 0.5,
                        "reasoning": "Unexpected model output"
                    }
                logger.info(f"🔧 Raw probabilities: SUPPORT={result['SUPPORT']:.3f}, REFUTE={result['REFUTE']:.3f}")
                
                # Normalize to ensure sum = 1.0
                total = sum(result.values())
                if total > 0:
                    for key in result:
                        result[key] = result[key] / total
                # DEBUG: Log after normalization
                logger.info(f"🔧 After normalization: SUPPORT={result['SUPPORT']:.3f}, REFUTE={result['REFUTE']:.3f}")
        
                
                logger.info(f"Model: SUPPORT:{result['SUPPORT']:.2f} REFUTE:{result['REFUTE']:.2f}")
                return result
            
        except Exception as e:
            logger.error(f"Model prediction error: {e}")
            return {
                "SUPPORT": 0.5,
                "REFUTE": 0.5,
                "reasoning": f"Model error: {str(e)}"
            }

    def get_transformer_explanation(self, original_text: str, prediction: Dict) -> Tuple[List[Dict], str]:
        """
        Generate explanation using Transformer Interpret for binary classification
        Uses ORIGINAL text for explanation generation (more natural for users)
        """
        try:
            if not self.transformer_explainer:
                logger.info("Transformer Interpret: Not available")
                return [], "Transformer explanation not available."
            
            # Map prediction to class name
            target_class = "SUPPORT" if prediction['prediction'] == "SUPPORT" else "REFUTE"
            
            logger.info(f"Transformer Interpret: Analyzing for '{target_class}' using original text")
            
            # Get word attributions using ORIGINAL text
            raw_attributions = self.transformer_explainer(
                original_text,
                class_name=target_class,
                internal_batch_size=1
            )
            
            # Process attributions (merge subwords, filter important tokens)
            processed_attributions = self._process_attributions(raw_attributions)
            
            # Generate explanation using ORIGINAL text
            explanation = self._format_binary_explanation(original_text, processed_attributions, prediction)
            
            return processed_attributions[:10], explanation
            
        except Exception as e:
            logger.error(f"Transformer Interpret ERROR: {e}")
            return [], f"Error generating explanation: {str(e)}"

    def _process_attributions(self, raw_attributions) -> List[Dict]:
        # Enhanced subword merging for RoBERTa tokenization
        merged_tokens = []
        merged_scores = []
        current_token = ""
        current_score = 0.0
        
        logger.debug(f"Raw attributions: {raw_attributions[:10]}")  # Debug first 10
        
        for token, score in raw_attributions:
            # Handle RoBERTa tokenization patterns
            if token.startswith("##"):
                # Standard BERT-style subword continuation
                current_token += token[2:]
                current_score += score
            elif token.startswith("Ġ"):
                # RoBERTa/GPT-style space prefix (start of new word)
                if current_token:
                    merged_tokens.append(current_token)
                    merged_scores.append(current_score)
                current_token = token[1:]  # Remove the Ġ prefix
                current_score = score
            elif len(token) == 1 and token.isalpha() and current_token:
                # Single character that might be part of previous word
                current_token += token
                current_score += score
            elif current_token and not token.startswith("<") and not token.startswith("["):
                # Continuation of current word (no special prefix)
                # Only if current_token exists and token isn't special
                if len(token) <= 3 and token.isalpha():  # Short fragments likely belong to previous word
                    current_token += token
                    current_score += score
                else:
                    # Start new word
                    merged_tokens.append(current_token)
                    merged_scores.append(current_score)
                    current_token = token
                    current_score = score
            else:
                # Save previous token if exists
                if current_token:
                    merged_tokens.append(current_token)
                    merged_scores.append(current_score)
                # Start new token
                current_token = token
                current_score = score
        
        # Add the last token
        if current_token:
            merged_tokens.append(current_token)
            merged_scores.append(current_score)
        
        logger.debug(f"After merging: {list(zip(merged_tokens[:10], merged_scores[:10]))}")
        
        # Clean up tokens - remove special tokens and very short fragments
        cleaned_tokens = []
        cleaned_scores = []
        
        for token, score in zip(merged_tokens, merged_scores):
            # Skip special tokens and very short fragments
            if (len(token) >= 2 and 
                not token.startswith("<") and 
                not token.startswith("[") and
                not token in [".", ",", "!", "?", ":", ";", "'", '"']):
                cleaned_tokens.append(token)
                cleaned_scores.append(score)
        
        logger.debug(f"After cleaning: {list(zip(cleaned_tokens[:10], cleaned_scores[:10]))}")
        
        # Create attribution list
        attributions = []
        for token, score in zip(cleaned_tokens, cleaned_scores):
            attributions.append({
                "word": token,
                "attribution": float(score),
                "importance": abs(float(score))
            })
        
        # Sort by importance
        attributions.sort(key=lambda x: x["importance"], reverse=True)
        
        logger.info(f"🔍 Processed attributions: {[(attr['word'], attr['attribution']) for attr in attributions[:6]]}")
        
        return attributions

    def _format_binary_explanation(self, original_text: str, attributions: List[Dict], prediction: Dict) -> str:
        """Generate user-friendly explanation for binary classification using original text"""
        try:
            if self.gemini_model and attributions:
                # Apply threshold filtering for meaningful words
                meaningful_threshold = 0.1  # Only words with |attribution| > 0.1
                meaningful_words = [
                    attr for attr in attributions 
                    if attr["importance"] > meaningful_threshold
                ]
                
                # If no words pass threshold, use top 3 most important
                if not meaningful_words:
                    meaningful_words = sorted(attributions, key=lambda x: x["importance"], reverse=True)[:3]
                    logger.info(f"No words passed threshold {meaningful_threshold}, using top 3")
                else:
                    logger.info(f"Found {len(meaningful_words)} words above threshold {meaningful_threshold}")
                
                # Limit to top 6 even after threshold filtering
                top_meaningful = meaningful_words[:6]
                
                words_str = ", ".join([
                    f"{attr['word']} ({attr['attribution']:+.2f})" 
                    for attr in top_meaningful
                ])
                
                classification = "TRUTHFUL" if prediction['prediction'] == "SUPPORT" else "MISINFORMATION"
                confidence = prediction['confidence']
                logger.info(f"Classification: {classification}, Confidence: {confidence * 100:.1f}%") 
                
                logger.info(f"Passing to Gemini: {words_str}")
                
                prompt = f"""
                The model classified the following statement as **{prediction['prediction']}** with {confidence * 100:.1f}% confidence.

                Statement: "{original_text}"

                Here are the key influential words and their influence scores on the model's prediction:
                {words_str}

                Note: Positive scores indicate words that support the predicted class, while negative scores indicate words that oppose it. The magnitude of the score reflects the strength of the influence.

                Using this information, briefly explain why the model likely made this prediction based on these words and their scores. 
                Ignore trivial or generic terms, and keep the explanation faithful to the sentence's meaning.

                Then, write a short follow-up sentence that adds helpful scientific context or clarification related to the topic.
                This may include a real-world implication, fact, or environmental insight that helps the user understand the statement more fully.
                Keep everything concise and accurate.
                """
                
                response = self.gemini_model.generate_content(prompt)
                return response.text.strip()
            
        except Exception as e:
            logger.error(f"Explanation generation error: {e}")
        
        # Fallback explanation
        classification = "truthful based on climate science evidence" if prediction['prediction'] == "SUPPORT" else "misinformation contradicted by climate science evidence"
        confidence = prediction['confidence']
        return f"This statement was classified as {classification} with {confidence* 100:.1f} confidence by our trained climate misinformation detection model."

    def get_explanation_and_attributions(self, original_text: str, prediction: Dict) -> Tuple[str, List[Dict]]:
        """Get comprehensive explanation and word attributions using original text"""
        word_attributions, explanation = self.get_transformer_explanation(original_text, prediction)
        return explanation, word_attributions

    def preprocess_for_model(self, original_text: str) -> Tuple[str, Dict[str, str]]:
        """
        Preprocess text for model input and return both processed text and debug info
        """
        try:
            # Apply preprocessing pipeline
            preprocessed_text = self.text_preprocessor.preprocess_text(original_text)
            
            # Create debug info
            preprocessing_info = {
                "original": original_text,
                "preprocessed": preprocessed_text,
                "preprocessing_applied": "Text normalization, lowercasing, stopword removal, lemmatization"
            }
            
            logger.info(f"📝 Preprocessing applied:")
            logger.info(f"   Original: '{original_text}'")
            logger.info(f"   Preprocessed: '{preprocessed_text}'")
            
            return preprocessed_text, preprocessing_info
            
        except Exception as e:
            logger.error(f"Preprocessing error: {e}")
            # Return original text if preprocessing fails
            return original_text, {
                "original": original_text,
                "preprocessed": original_text,
                "preprocessing_applied": f"Error during preprocessing: {str(e)}"
            }


# API Endpoints
@app.get("/")
async def root():
    return {"message": "Climate Misinformation Detection API v2.1 - With Preprocessing", "status": "running"}

@app.get("/health")
async def health_check():
    global pipeline
    
    if not pipeline:
        return {"status": "initializing"}
    
    return {
        "status": "healthy",
        "components": {
            "gemini": pipeline.gemini_model is not None,
            "binary_model": pipeline.trained_model is not None,
            "transformer_interpret": pipeline.transformer_explainer is not None,
            "transformer_interpret_available": TRANSFORMER_INTERPRET_AVAILABLE,
            "text_preprocessor": pipeline.text_preprocessor is not None
        },
        "version": "2.1 - With Text Preprocessing Pipeline"
    }

@app.post("/detect", response_model=DetectionResult)
async def detect_misinformation(request: QueryRequest):
    """Main detection endpoint with preprocessing pipeline"""
    try:
        global pipeline
        
        if not pipeline:
            raise HTTPException(status_code=503, detail="Pipeline not ready")
        
        original_text = request.text.strip()
        
        if not original_text:
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        logger.info(f"🎯 Analyzing: {original_text}")
        
        # Stage 1 & 2: Domain check + Completeness check using ORIGINAL text
        combined_result = pipeline.check_climate_domain_and_completeness(original_text)
        
        if not combined_result["is_climate_related"]:
            return DetectionResult(
                is_climate_related=False,
                status="out_of_domain"
            )
        
        if not combined_result["is_complete"]:
            return DetectionResult(
                is_climate_related=True,
                is_complete_statement=False,
                completeness_guidance=combined_result["guidance"],
                status="incomplete_statement"
            )
        
        # Stage 3: Preprocess text for model input
        preprocessed_text, preprocessing_info = pipeline.preprocess_for_model(original_text)
        
        # Stage 4: Model prediction using PREPROCESSED text
        model_prediction = pipeline.get_model_prediction(preprocessed_text)
        # DEBUG: Log model prediction
        logger.info(f"🔧 Model prediction: {model_prediction}")
        
        # Use only model prediction as final result
        final_result = {
            "prediction": "SUPPORT" if model_prediction["SUPPORT"] > model_prediction["REFUTE"] else "REFUTE",
            "confidence": float(max(model_prediction["SUPPORT"], model_prediction["REFUTE"])),
            "support_score": float(model_prediction["SUPPORT"]),
            "refute_score": float(model_prediction["REFUTE"])
        }
        # DEBUG: Log final result
        logger.info(f"🔧 Final result: {final_result}")
        logger.info(f"🔧 Final confidence: {final_result['confidence']}")
        
        # Stage 5: Get explanations and word attributions using ORIGINAL text
        explanation, word_attributions = pipeline.get_explanation_and_attributions(original_text, final_result)
        
        return DetectionResult(
            is_climate_related=True,
            is_complete_statement=True,
            model_prediction=model_prediction,
            final_result=final_result,
            explanation=explanation,
            confidence=final_result["confidence"],
            word_attributions=word_attributions,
            status="complete_analysis",
            preprocessing_info=preprocessing_info  # Include preprocessing debug info
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Detection error: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")

@app.on_event("startup")
async def startup_event():
    global pipeline
    try:
        logger.info("Starting Climate Misinformation Detection API v2.1 - With Preprocessing...")
        
        # Initialize NLTK data during startup (after app is ready)
        initialize_nltk()
        
        # Initialize the pipeline
        pipeline = ClimateDetectionPipeline()
        logger.info("API ready!")
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        # Continue even if some components fail
        logger.info("API starting with limited functionality...")

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting server on http://localhost:8000")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)