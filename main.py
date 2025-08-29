from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
import os
import re
import openai
import requests
import time
import pandas as pd
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

# ONNX Runtime import
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("Warning: onnxruntime not available. Install with: pip install onnxruntime")

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
    allow_origins=["https://yourdomain.com", "https://cmi-with-xai-delicate-sun-4817.fly.dev"] ,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from transformers import AutoModel, AutoConfig, PreTrainedModel
import torch
import torch.nn as nn

class QueryRequest(BaseModel):
    text: str

class DetectionResult(BaseModel):
    is_climate_related: bool
    model_prediction: Optional[Dict[str, Any]] = None
    final_result: Optional[Dict[str, Any]] = None
    explanation: Optional[str] = None
    confidence: Optional[float] = None
    word_attributions: Optional[List[Dict[str, Any]]] = None
    status: str  # "complete_analysis" or "out_of_domain"
    # Debug info (optional)
    preprocessing_info: Optional[Dict[str, str]] = None

# Global variables
pipeline = None
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_PATH = os.getenv("MODEL_PATH")  
ONNX_MODEL_PATH = os.getenv("ONNX_MODEL_PATH")  # For fast inference
PYTORCH_MODEL_PATH = os.getenv("PYTORCH_MODEL_PATH")  # For XAI
USE_ONNX_FOR_INFERENCE = os.getenv("USE_ONNX_FOR_INFERENCE", "true").lower() == "true"
QUANTIZE_MODEL_PATH = os.getenv("QUANTIZE_MODEL_PATH")  # For quantized model

class TextPreprocessor:
    """
    Minimal text preprocessing pipeline that preserves semantic meaning
    """
    def __init__(self):
        """
        Very minimal preprocessor that preserves semantic meaning
        """
        pass

    def minimal_clean_text(self, text: str) -> str:
        """
        Super minimal text cleaning - just basic normalization
        
        Args:
            text (str): Raw text to clean
            
        Returns:
            str: Minimally cleaned text
        """
        if pd.isna(text) or text == '':
            return ''

        # Convert to string if not already
        text = str(text)

        # 1. Remove URLs only
        text = re.sub(r'http[s]?://\S+', ' ', text)
        text = re.sub(r'www\.\S+', ' ', text)

        # 2. Clean hashtags - remove # but keep the word
        text = re.sub(r'#(\w+)', r'\1', text)

        # 3. Remove @mentions
        text = re.sub(r'@\w+', ' ', text)

        # 4. Normalize whitespace only
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n+', ' ', text)

        # 5. Convert to lowercase
        text = text.lower()

        # 6. Remove extra spaces
        text = ' '.join(text.split())

        return text.strip()
    
    def preprocess_text(self, text: str) -> str:
        """
        Complete minimal preprocessing pipeline
        """
        return self.minimal_clean_text(text)

class ClimateDetectionPipeline:
    def __init__(self):
        self.gemini_model = None
        self.gpt_client = None
        self.groq_client = None
        
        # Dual model setup
        self.onnx_model = None
        self.pytorch_model = None
        self.tokenizer = None
        self.transformer_explainer = None
        
        # Initialize text preprocessor with minimal processing
        self.text_preprocessor = TextPreprocessor()
        
        logger.info("Initializing Climate Detection Pipeline...")
        
        # Setup all LLM services
        self.setup_all_llms()
        self.load_dual_models()
        self.setup_xai()
        
        logger.info("Pipeline initialization complete!")

    def setup_all_llms(self):
        """Initialize all available LLM services"""
        self.setup_gemini()
        
        # Log which services are available
        available_services = []
        if self.gemini_model: available_services.append("Gemini")
        if self.groq_client: available_services.append("LLaMA-3")
        
        logger.info(f"Available LLM services: {', '.join(available_services) if available_services else 'None'}")

    def load_dual_models(self):
        """Load both ONNX (for speed) and PyTorch (for XAI) models"""
        try:
            logger.info(" Loading dual model setup...")
            
            # Determine tokenizer path
            tokenizer_path = None
            if PYTORCH_MODEL_PATH and os.path.exists(PYTORCH_MODEL_PATH):
                tokenizer_path = PYTORCH_MODEL_PATH
            elif ONNX_MODEL_PATH and os.path.exists(os.path.dirname(ONNX_MODEL_PATH)):
                tokenizer_path = os.path.dirname(ONNX_MODEL_PATH)
            elif MODEL_PATH and os.path.exists(MODEL_PATH):
                tokenizer_path = MODEL_PATH
            else:
                raise FileNotFoundError("No valid model path found for tokenizer")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained("ImaanIbrar/Climate-misinformation-classification")
            logger.info(" Tokenizer loaded")
            
            # Load ONNX model for fast inference
            if ONNX_MODEL_PATH and os.path.exists(ONNX_MODEL_PATH) and ONNX_AVAILABLE:
                self._load_onnx_model()
            else:
                if not ONNX_AVAILABLE:
                    logger.warning("ONNX Runtime not available - install with: pip install onnxruntime")
                else:
                    logger.warning("ONNX model not found - will use PyTorch for inference")
            
            # Load PyTorch model for XAI
            if PYTORCH_MODEL_PATH and os.path.exists(PYTORCH_MODEL_PATH):
                self._load_pytorch_model()
            elif MODEL_PATH and os.path.exists(MODEL_PATH):
                # Fallback to single model setup
                logger.info("Using single model path for PyTorch model")
                self._load_pytorch_model_fallback()
            else:
                logger.warning("PyTorch model not found - XAI features disabled")
            
            # Validate at least one model is loaded
            if not self.onnx_model and not self.pytorch_model:
                raise FileNotFoundError("Neither ONNX nor PyTorch model could be loaded")
            
            logger.info("Dual model setup complete!")
            
        except Exception as e:
            logger.error(f" Failed to load models: {e}")
            raise e

    def _load_onnx_model(self):
        """Load ONNX model for fast inference"""
        try:
            logger.info(" Loading ONNX model for faster inference...")
            
            providers = ['CPUExecutionProvider']
            # Uncomment next line if you have GPU: providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            
            self.onnx_model = ort.InferenceSession(ONNX_MODEL_PATH, providers=providers)
            
            # Test ONNX model
            self._test_onnx_model()
            logger.info("✅ ONNX model loaded and tested successfully")
            
        except Exception as e:
            logger.error(f"❌ ONNX model loading failed: {e}")
            self.onnx_model = None

    def _load_pytorch_model(self):
        """Load PyTorch model for XAI"""
        try:
            logger.info("🔬 Loading PyTorch model for XAI...")
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                self.pytorch_model = AutoModelForSequenceClassification.from_pretrained("ImaanIbrar/Climate-misinformation-classification")
                # self.pytorch_model = torch.load(QUANTIZE_MODEL_PATH)
                self.pytorch_model.eval()
            
            # Test PyTorch model
            self._test_pytorch_model()
            logger.info("PyTorch model loaded and tested successfully")
            
        except Exception as e:
            logger.error(f"PyTorch model loading failed: {e}")
            self.pytorch_model = None

    def _load_pytorch_model_fallback(self):
        """Load PyTorch model from fallback MODEL_PATH"""
        try:
            logger.info("Loading PyTorch model from fallback path...")
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.pytorch_model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
                self.pytorch_model.eval()
            
            # Test PyTorch model
            self._test_pytorch_model()
            logger.info("PyTorch model loaded and tested successfully (fallback)")
            
        except Exception as e:
            logger.error(f"PyTorch model loading failed (fallback): {e}")
            self.pytorch_model = None

    def _test_onnx_model(self):
        """Test ONNX model"""
        test_text = "test climate change"
        inputs = self.tokenizer(test_text, return_tensors="np", truncation=True, padding=True, max_length=256)
        
        ort_inputs = {
            'input_ids': inputs['input_ids'].astype(np.int64),
            'attention_mask': inputs['attention_mask'].astype(np.int64)
        }
        
        outputs = self.onnx_model.run(None, ort_inputs)
        logits = outputs[0]
        
        logger.info(f"ONNX Model verification - Output shape: {logits.shape}")
        if logits.shape[-1] != 2:
            raise ValueError(f"Expected 2 classes, but ONNX model outputs {logits.shape[-1]} classes")

    def _test_pytorch_model(self):
        """Test PyTorch model"""
        test_input = self.tokenizer("test climate change", return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.pytorch_model(**test_input)
            logits = outputs.logits
            
            logger.info(f"PyTorch Model verification - Output shape: {logits.shape}")
            if logits.shape[-1] != 2:
                raise ValueError(f"Expected 2 classes, but PyTorch model outputs {logits.shape[-1]} classes")


    def setup_gemini(self):
        """Initialize Gemini API with correct model name"""
        try:
            logger.info("Setting up Gemini...")
            
            if not GEMINI_API_KEY or GEMINI_API_KEY == "your-gemini-api-key-here":
                logger.warning("GEMINI_API_KEY not set")
                return
            
            genai.configure(api_key=GEMINI_API_KEY)
            
            model_names = [
                'gemini-2.5-flash'
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
    
    def setup_groq(self):
        """Initialize Groq client and verify LLaMA 3 model connection"""
        try:
            logger.info("Setting up Groq...")
            from groq import Groq

            if not GROQ_API_KEY:
                logger.warning("GROQ_API_KEY not set")
                self.groq_client = None
                return

            self.groq_client = Groq(api_key=GROQ_API_KEY)

            # Test connection
            test_response = self.groq_client.chat.completions.create(
                model="llama3-70b-8192",
                messages=[{"role": "user", "content": "Say 'test'"}],
                temperature=0,
                max_completion_tokens=50,
                stream=False
            )

            logger.info("Groq LLaMA-3 setup successful")
        except ImportError:
            logger.error("groq package not installed. Run: pip install groq")
            self.groq_client = None
        except Exception as e:
            logger.error(f"Groq setup failed: {e}")
            self.groq_client = None

    def setup_xai(self):
        """Setup XAI components - Requires PyTorch model"""
        try:
            if not self.pytorch_model:
                logger.warning("XAI features disabled - PyTorch model not available")
                self.transformer_explainer = None
                return
            
            logger.info("Setting up XAI components...")
            
            if TRANSFORMER_INTERPRET_AVAILABLE:
                try:
                    self.transformer_explainer = SequenceClassificationExplainer(
                        self.pytorch_model, 
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

    def build_domain_prompt(self, original_text: str) -> str:
        """Build a refined prompt for climate domain checking only"""
        return f"""You are evaluating whether a statement is related to climate or environmental science.

    ==========================
    DOMAIN CHECK
    ==========================
    Is the statement about climate, climate change, environmental science, or related topics?
    
    Relevant domains include:
    - Greenhouse gases (e.g. CO₂, CH₄), global warming, temperature rise
    - Fossil fuels, renewable energy, carbon footprint, emissions
    - Trees, forests, biodiversity, ecosystems, deforestation
    - Sea level rise, melting glaciers, ice caps, ocean acidification
    - Extreme weather (droughts, hurricanes, floods), wildfires
    - Human impact on ecosystems, climate policy, environmental regulations
    - Weather patterns, atmospheric changes, ozone layer
    - Renewable vs non-renewable energy sources
    - Environmental pollution, air quality, water quality
    
    Statements that are vague, off-topic, or emotional without any clear connection to climate/environment (e.g., "I love you climate") should be marked as **NOT_CLIMATE_RELATED**.

    If yes, mark as: DOMAIN: CLIMATE_RELATED  
    If not, mark as: DOMAIN: NOT_CLIMATE_RELATED

    ==========================
    NOW ANALYZE THIS STATEMENT:
    ==========================

    Statement: "{original_text}"

    REQUIRED RESPONSE FORMAT:
    DOMAIN: [CLIMATE_RELATED or NOT_CLIMATE_RELATED]  
    REASONING: [Brief explanation of why this is or isn't climate-related]
    """

    def parse_domain_response(self, response_text: str) -> Dict[str, Any]:
        """Parse domain response from LLM models"""
        lines = response_text.strip().split('\n')
        
        # More flexible parsing to handle different formatting
        domain_line = ""
        reasoning_line = ""
        
        for line in lines:
            line_clean = line.strip().replace('*', '').replace('#', '').strip()
            if line_clean.upper().startswith('DOMAIN:'):
                domain_line = line_clean
            elif line_clean.upper().startswith('REASONING:'):
                reasoning_line = line_clean

        # Fix the parsing logic - check for NOT_CLIMATE_RELATED first
        domain_upper = domain_line.upper()
        if 'NOT_CLIMATE_RELATED' in domain_upper:
            is_climate_related = False
        elif 'CLIMATE_RELATED' in domain_upper:
            is_climate_related = True
        else:
            # Fallback parsing - look for keywords in the entire response
            response_upper = response_text.upper()
            if 'NOT_CLIMATE_RELATED' in response_upper or 'NOT CLIMATE' in response_upper:
                is_climate_related = False
            elif 'CLIMATE_RELATED' in response_upper or 'CLIMATE RELATED' in response_upper:
                is_climate_related = True
            else:
                # Default to false if unclear
                logger.warning(f"Unclear domain response, defaulting to NOT_CLIMATE_RELATED: {response_text[:100]}")
                is_climate_related = False

        reasoning = reasoning_line.replace('REASONING:', '').strip() if reasoning_line else "Domain analysis completed"

        logger.info(f"Domain check result: {'CLIMATE_RELATED' if is_climate_related else 'NOT_CLIMATE_RELATED'}")
        if reasoning:
            logger.info(f"Reasoning: {reasoning}")

        return {
            "is_climate_related": is_climate_related,
            "reasoning": reasoning
        }

    def default_domain_response(self, is_climate: bool = True) -> Dict[str, Any]:
        """Default response when domain check fails"""
        return {
            "is_climate_related": is_climate,
            "reasoning": "Error in domain check - proceeding with default classification"
        }

    def check_climate_domain_GPT4o(self, original_text: str) -> Dict[str, Any]:
        """Uses OpenAI GPT-4o to evaluate climate domain"""
        try:
            if not self.gpt_client:
                logger.error("GPT client not available")
                return self.default_domain_response()

            prompt = self.build_domain_prompt(original_text)
            messages = [{"role": "user", "content": prompt}]
            start = time.time()

            response = self.gpt_client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=0,
                max_tokens=256,
                top_p=1
            )

            end = time.time()
            content = response.choices[0].message.content.strip()

            logger.info(f"[GPT-4o] Response time: {end - start:.2f} seconds")
            logger.info(f"[GPT-4o] Raw response: {content}")

            return self.parse_domain_response(content)

        except Exception as e:
            logger.error(f"[GPT-4o] Error: {e}")
            return self.default_domain_response()

    def check_climate_domain_LLaMA(self, original_text: str) -> Dict[str, Any]:
        """Uses Groq's LLaMA-3 70B model to evaluate climate domain"""
        try:
            if not self.groq_client:
                logger.error("Groq client not available")
                return self.default_domain_response()

            prompt = self.build_domain_prompt(original_text)
            messages = [{"role": "user", "content": prompt}]
            start = time.time()

            response = self.groq_client.chat.completions.create(
                model="llama3-70b-8192",
                messages=messages,
                temperature=0,
                max_completion_tokens=256,
                top_p=1,
                stream=False
            )

            end = time.time()
            content = response.choices[0].message.content.strip()

            logger.info(f"[LLaMA-3 Groq] Response time: {end - start:.2f} seconds")
            logger.info(f"[LLaMA-3 Groq] Raw response: {content}")

            return self.parse_domain_response(content)

        except Exception as e:
            logger.error(f"[LLaMA-3 Groq] Error: {e}")
            return self.default_domain_response()

    def check_climate_domain_Gemini(self, original_text: str) -> Dict[str, Any]:
        """Uses Gemini model for domain check"""
        try:
            if not self.gemini_model:
                logger.error("Gemini model not available")
                return self.default_domain_response()

            prompt = self.build_domain_prompt(original_text)
            start = time.time()

            response = self.gemini_model.generate_content(prompt)
            response_text = response.text.strip()

            end = time.time()
            logger.info(f"[Gemini] Response time: {end - start:.2f} seconds")
            logger.info(f"[Gemini] Raw response: {response_text}")

            return self.parse_domain_response(response_text)

        except Exception as e:
            logger.error(f"[Gemini] Error: {e}")
            return self.default_domain_response()

    def check_climate_domain_keyword_fallback(self, original_text: str) -> Dict[str, Any]:
        """Keyword-based fallback when no LLM is available"""
        logger.info("Using keyword-based fallback")
        climate_keywords = [
            'climate', 'warming', 'carbon', 'temperature', 'greenhouse', 
            'emissions', 'co2', 'fossil', 'renewable', 'environment',
            'pollution', 'weather', 'arctic', 'ice', 'ocean', 'atmosphere',
            'sea level', 'glacier', 'wildfire', 'drought', 'flood', 'ozone',
            'deforestation', 'biodiversity', 'ecosystem', 'sustainability'
        ]
        text_lower = original_text.lower()
        is_climate = any(keyword in text_lower for keyword in climate_keywords)
        
        return {
            "is_climate_related": is_climate,
            "reasoning": "Keyword-based domain classification - No LLM services available"
        }

    def check_climate_domain(self, original_text: str, model_name: str = "gemini") -> Dict[str, Any]:
        """
        Main function to check climate domain using specified model
        
        Args:
            original_text: The text to analyze
            model_name: Which model to use ("gpt4o", "gemini", "llama", "keyword")
        
        Returns:
            Dict with domain analysis results
        """
        logger.info(f"🔍 Using model: {model_name.upper()}")
        
        # Route to specific model functions
        if model_name.lower() in ["gpt4o", "gpt"]:
            return self.check_climate_domain_GPT4o(original_text)
        elif model_name.lower() == "gemini":
            return self.check_climate_domain_Gemini(original_text)
        elif model_name.lower() in ["llama", "llama3"]:
            return self.check_climate_domain_LLaMA(original_text)
        elif model_name.lower() == "keyword":
            return self.check_climate_domain_keyword_fallback(original_text)
        else:
            logger.error(f"Invalid model name: {model_name}")
            logger.info("Available models: gpt4o, gemini, llama, keyword")
            return self.default_domain_response()

    def get_model_prediction(self, preprocessed_text: str) -> Dict[str, Any]:
        """
        Get binary model prediction - Uses ONNX for speed, falls back to PyTorch
        """
        try:
            if not self.tokenizer:
                return {
                    "True": 0.5,
                    "False": 0.5,
                    "reasoning": "No tokenizer available"
                }
            
            logger.info(f"Model input (preprocessed): '{preprocessed_text}'")
            
            # Use ONNX for fast inference if available and enabled
            if self.onnx_model and USE_ONNX_FOR_INFERENCE:
                return self._predict_onnx(preprocessed_text)
            # Fall back to PyTorch
            elif self.pytorch_model:
                return self._predict_pytorch(preprocessed_text)
            else:
                return {
                    "True": 0.5,
                    "False": 0.5,
                    "reasoning": "No model available for inference"
                }
                
        except Exception as e:
            logger.error(f"Model prediction error: {e}")
            return {
                "True": 0.5,
                "False": 0.5,
                "reasoning": f"Model error: {str(e)}"
            }

    def _predict_onnx(self, preprocessed_text: str) -> Dict[str, Any]:
        """ONNX inference - FASTER"""
        try:
            inputs = self.tokenizer(
                preprocessed_text, 
                return_tensors="np",
                truncation=True, 
                padding=True,
                max_length=256
            )
            
            ort_inputs = {
                'input_ids': inputs['input_ids'].astype(np.int64),
                'attention_mask': inputs['attention_mask'].astype(np.int64)
            }
            
            start_time = time.time()
            outputs = self.onnx_model.run(None, ort_inputs)
            inference_time = time.time() - start_time
            
            logits = outputs[0]
            probs = self._softmax(logits)
            
            if probs.ndim > 1:
                probs = probs.squeeze()
            
            logger.info(f" ONNX inference time: {inference_time*1000:.2f}ms")
            
            return self._process_predictions(probs)
            
        except Exception as e:
            logger.error(f"ONNX prediction error: {e}")
            # Fall back to PyTorch if ONNX fails
            if self.pytorch_model:
                logger.info("Falling back to PyTorch for this prediction")
                return self._predict_pytorch(preprocessed_text)
            else:
                raise e

    def _predict_pytorch(self, preprocessed_text: str) -> Dict[str, Any]:
        """PyTorch inference - SLOWER but supports XAI"""
        try:
            inputs = self.tokenizer(
                preprocessed_text, 
                return_tensors="pt", 
                truncation=True, 
                padding=True,
                max_length=256
            )

            decoded = self.tokenizer.decode(inputs['input_ids'][0])
            logger.info(f" Tokenized and decoded back: '{decoded}'")
            
            with torch.no_grad():
                start_time = time.time()
                outputs = self.pytorch_model(**inputs)
                inference_time = time.time() - start_time
                
                logits = outputs.logits
                probs = torch.nn.functional.softmax(logits, dim=-1)
                
                if probs.dim() > 1:
                    probs = probs.squeeze()
                
                probs_np = probs.cpu().numpy()
                
                logger.info(f"PyTorch inference time: {inference_time*1000:.2f}ms")
                
                return self._process_predictions(probs_np)
                
        except Exception as e:
            logger.error(f"PyTorch prediction error: {e}")
            raise e

    def _softmax(self, x):
        """Softmax function for numpy arrays"""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def _process_predictions(self, probs_np) -> Dict[str, Any]:
        """Process predictions (same for both ONNX and PyTorch)"""
        if probs_np.ndim > 1:
            probs_np = probs_np[0]
        
        if len(probs_np) == 2:
            result = {
                "True": float(probs_np[1]),
                "False": float(probs_np[0])
            }
        elif len(probs_np) == 3:
            support_score = float(probs_np[1])
            refute_score = float(probs_np[0])
            neutral_score = float(probs_np[2])
            
            if support_score >= refute_score:
                support_score += neutral_score * 0.3
                refute_score += neutral_score * 0.7
            else:
                support_score += neutral_score * 0.7
                refute_score += neutral_score * 0.3
            
            result = {
                "True": support_score,
                "False": refute_score
            }
        else:
            logger.warning(f"Unexpected model output size: {len(probs_np)}")
            return {
                "True": 0.5,
                "False": 0.5,
                "reasoning": "Unexpected model output"
            }
        
        # Normalize
        total = sum(result.values())
        if total > 0:
            for key in result:
                result[key] = result[key] / total
        
        logger.info(f" Model: True:{result['True']:.2f} False:{result['False']:.2f}")
        return result

    def get_transformer_explanation(self, original_text: str, prediction: Dict) -> Tuple[List[Dict], str]:
        """
        Generate explanation using Transformer Interpret - Uses PyTorch model
        """
        try:
            if not self.transformer_explainer or not self.pytorch_model:
                logger.info(" Transformer Interpret: Not available (PyTorch model required)")
                return [], "XAI explanation not available - PyTorch model required for explanations."
            
            # Map prediction to class name
            target_class = "True" if prediction['prediction'] == "True" else "False"
            
            logger.info(f"Transformer Interpret: Analyzing for '{target_class}' using PyTorch model")
            
            # Get word attributions using PyTorch model
            raw_attributions = self.transformer_explainer(
                original_text,
                class_name=target_class,
                internal_batch_size=1
            )
            
            # Process attributions
            processed_attributions = self._process_attributions(raw_attributions)
            
            # Generate explanation  
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
        
        logger.info(f" Processed attributions: {[(attr['word'], attr['attribution']) for attr in attributions[:6]]}")
        
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
                
                classification = "TRUTHFUL" if prediction['prediction'] == "True" else "MISINFORMATION"
                confidence = prediction['confidence']
                logger.info(f" Classification: {classification}, Confidence: {confidence * 100:.1f}%") 
                start = time.time()
                logger.info(f" Passing to Gemini: {words_str}")
                
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
                end = time.time()
                logger.info(f"✨ Gemini explanation time: {end - start:.2f} seconds")
                return response.text.strip()
            
        except Exception as e:
            logger.error(f"Explanation generation error: {e}")
        
        # Fallback explanation
        classification = "truthful based on climate science evidence" if prediction['prediction'] == "True" else "misinformation contradicted by climate science evidence"
        confidence = prediction['confidence']
        return f"This statement was classified as {classification} with {confidence* 100:.1f}% confidence by our trained climate misinformation detection model."

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
                "preprocessing_applied": "Minimal cleaning: URL removal, hashtag cleaning, whitespace normalization, lowercase conversion - NO tokenization or word removal"
            }
            
            logger.info(f"🔧 Preprocessing applied:")
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
    return {"message": "Climate Misinformation Detection API v3.0 - Minimal Preprocessing", "status": "running"}

@app.get("/health")
async def health_check():
    global pipeline
    
    if not pipeline:
        return {"status": "initializing"}
    
    return {
        "status": "healthy",
        "components": {
            "gpt4o": pipeline.gpt_client is not None,
            "gemini": pipeline.gemini_model is not None,
            "llama": pipeline.groq_client is not None,
            "onnx_model": pipeline.onnx_model is not None,
            "pytorch_model": pipeline.pytorch_model is not None,
            "transformer_interpret": pipeline.transformer_explainer is not None,
            "transformer_interpret_available": TRANSFORMER_INTERPRET_AVAILABLE,
            "onnx_available": ONNX_AVAILABLE,
            "text_preprocessor": pipeline.text_preprocessor is not None
        },
        "inference_mode": "ONNX" if (pipeline.onnx_model and USE_ONNX_FOR_INFERENCE) else "PyTorch",
        "version": "3.0 - Minimal Preprocessing (ONNX + PyTorch)",
        "preprocessing": "Minimal cleaning only - preserves semantic meaning"
    }

@app.post("/detect", response_model=DetectionResult)
async def detect_misinformation(request: QueryRequest):
    """Main detection endpoint with minimal preprocessing"""
    try:
        global pipeline
        
        if not pipeline:
            raise HTTPException(status_code=503, detail="Pipeline not ready")
        
        original_text = request.text.strip()
        
        if not original_text:
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        logger.info(f" Analyzing: {original_text}")
        
        # ============================================
        # CHANGE MODEL HERE - Just modify this line:
        # ============================================
        MODEL_TO_USE = "gemini"  # Options: "gpt4o", "gemini", "llama", "keyword"
        complete_start_time = time.time()
        
        # Stage 1: Domain check using specified model
        domain_result = pipeline.check_climate_domain(original_text, model_name=MODEL_TO_USE)
        
        if not domain_result["is_climate_related"]:
            return DetectionResult(
                is_climate_related=False,
                status="out_of_domain"
            )
        
        # Stage 2: Preprocess text for model input (MINIMAL PREPROCESSING)
        preprocessed_text, preprocessing_info = pipeline.preprocess_for_model(original_text)
        
        # Stage 3: Model prediction using PREPROCESSED text (ONNX or PyTorch)
        start_model = time.time()
        model_prediction = pipeline.get_model_prediction(preprocessed_text)
        end_model = time.time()
        
        inference_mode = "ONNX" if (pipeline.onnx_model and USE_ONNX_FOR_INFERENCE) else "PyTorch"
        logger.info(f" {inference_mode} prediction time: {end_model - start_model:.2f} seconds")
        logger.info(f" Model prediction: {model_prediction}")
        
        # Use only model prediction as final result
        final_result = {
            "prediction": "True" if model_prediction["True"] > model_prediction["False"] else "False",
            "confidence": float(max(model_prediction["True"], model_prediction["False"])),
            "support_score": float(model_prediction["True"]),
            "refute_score": float(model_prediction["False"])
        }
        
        # Stage 4: Get explanations and word attributions using ORIGINAL text (requires PyTorch)
        start_expl = time.time()
        explanation, word_attributions = pipeline.get_explanation_and_attributions(original_text, final_result)
        end_expl = time.time()
        logger.info(f" Explanation generation time: {end_expl - start_expl:.2f} seconds")
        complete_end_time = time.time()
        logger.info(f" Total time: {complete_end_time - complete_start_time:.2f} seconds")
        
        return DetectionResult(
            is_climate_related=True,
            model_prediction=model_prediction,
            final_result=final_result,
            explanation=explanation,
            confidence=final_result["confidence"],
            word_attributions=word_attributions,
            status="complete_analysis",
            preprocessing_info=preprocessing_info
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
        logger.info(" Starting Climate Misinformation Detection API v3.0 - Minimal Preprocessing...")
        
        # Initialize the pipeline
        pipeline = ClimateDetectionPipeline()
        logger.info(" API ready!")
    except Exception as e:
        logger.error(f" Startup failed: {e}")
        # Continue even if some components fail
        logger.info(" API starting with limited functionality...")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=False)