from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
import os
from typing import Dict, List, Any, Optional
import logging
from dotenv import load_dotenv
from typing import List, Tuple, Dict

import torch
import warnings
import numpy as np
from lime.lime_text import LimeTextExplainer
import google.generativeai as genai
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)

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
    gemini_prediction: Optional[Dict[str, Any]] = None
    model_prediction: Optional[Dict[str, Any]] = None
    final_result: Optional[Dict[str, Any]] = None
    explanation: Optional[str] = None
    confidence: Optional[float] = None

# Global variables
pipeline = None
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_PATH = os.getenv("MODEL_PATH")

class ClimateDetectionPipeline:
    def __init__(self):
        self.gemini_model = None
        self.trained_model = None
        self.tokenizer = None
        self.lime_explainer = None
        
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
            logger.info("Loading trained model...")

            if not os.path.exists(MODEL_PATH):
                logger.error(f"Model path not found: {MODEL_PATH}")
                raise FileNotFoundError

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
                self.trained_model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
                self.trained_model.eval()

            # Test the model and check output shape
            test_input = self.tokenizer("test climate change", return_tensors="pt", truncation=True, padding=True)
            with torch.no_grad():
                outputs = self.trained_model(**test_input)
                logits = outputs.logits
                probs = torch.nn.functional.softmax(logits, dim=-1)

                logger.info(f"Model loaded - Output shape: {probs.shape}")
                logger.info(f"Number of classes: {probs.shape[-1]}")

                if probs.shape[-1] != 3:
                    logger.warning(f"Model outputs {probs.shape[-1]} classes, expected 3")

        except Exception as e:
            logger.error(f"Failed to load fine-tuned model: {e}")
            logger.info("Falling back to default Roberta base model with 3 classes.")

            # Load Roberta base with 3 labels (not fine-tuned)
            self.tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-base")
            self.trained_model = AutoModelForSequenceClassification.from_pretrained("FacebookAI/roberta-base", num_labels=3)
            self.trained_model.eval()

            # Optional: Log the shape for the fallback model
            test_input = self.tokenizer("test climate change", return_tensors="pt", truncation=True, padding=True)
            with torch.no_grad():
                outputs = self.trained_model(**test_input)
                logits = outputs.logits
                probs = torch.nn.functional.softmax(logits, dim=-1)

                logger.info(f"Fallback model loaded - Output shape: {probs.shape}")
                logger.info(f"Number of classes: {probs.shape[-1]}")


    def setup_xai(self):
        """Setup XAI components with proper error handling"""
        try:
            if not self.trained_model:
                logger.warning("Skipping XAI setup - model not loaded")
                return
            
            logger.info("Setting up XAI components...")
            
            # Setup LIME with error handling
            try:
                self.lime_explainer = LimeTextExplainer(
                    class_names=['SUPPORT', 'REFUTE', 'NOT_ENOUGH_INFO']
                )
                logger.info("LIME initialized")
            except ImportError:
                logger.error("LIME not available")
            except Exception as e:
                logger.error(f"LIME setup failed: {e}")
            
        except Exception as e:
            logger.error(f"XAI setup failed: {e}")

    def get_default_scores(self, reason="Default"):
        """Return default scores"""
        return {
            "SUPPORT": 0.33,
            "REFUTE": 0.33,
            "NOT_ENOUGH_INFO": 0.34,
            "reasoning": reason
        }

    def check_climate_domain(self, text: str) -> bool:
        """Check if text is climate-related"""
        try:
            if not self.gemini_model:
                # Keyword-based fallback
                climate_keywords = [
                    'climate', 'warming', 'carbon', 'temperature', 'greenhouse', 
                    'emissions', 'co2', 'fossil', 'renewable', 'environment',
                    'pollution', 'weather', 'arctic', 'ice', 'ocean', 'atmosphere'
                ]
                text_lower = text.lower()
                is_climate = any(keyword in text_lower for keyword in climate_keywords)
                logger.info(f"Keyword-based domain check: {'YES' if is_climate else 'NO'}")
                return is_climate
            
            prompt = prompt = f"""Determine whether the following statement is related to climate or climate change.This may include — but is not limited to — topics such as: global warming, greenhouse gases, CO₂ emissions, temperature rise, melting ice, sea level rise, fossil fuels, deforestation, extreme weather events, renewable energy, or other environmental changes linked to the Earth's climate.Respond with only YES or NO.Statement: {text}"""

            response = self.gemini_model.generate_content(prompt)
            result = "YES" in response.text.upper()
            logger.info(f"Gemini domain check: {'YES' if result else 'NO'}")
            return result
            
        except Exception as e:
            logger.error(f"Domain check error: {e}")
            return True

    def get_gemini_prediction(self, text: str) -> Dict[str, Any]:
        """Get Gemini prediction with fallback"""
        try:
            if not self.gemini_model:
                return self.get_default_scores("Gemini unavailable")
            
            prompt = f"""You are tasked with evaluating the scientific credibility of the following statement related to climate change.

            Use only verified and authoritative sources — such as peer-reviewed research, consensus reports (e.g., IPCC, NASA, NOAA), or reputable scientific journalism — to assess the claim.

            Do not rely on general internet content, public opinion, blog posts, or anecdotal evidence. Your analysis should reflect only what is verifiable and recognized within established climate science.

            Return scores between 0.0 and 1.0 for each category below, such that the total sums to exactly 1.0:

            ---

            Example 1:  
            Statement: "Carbon dioxide levels have remained stable since 1900."  
            SUPPORT: Scientific evidence supports this statement  
            REFUTE: Scientific evidence contradicts this statement  
            NOT_ENOUGH_INFO: Insufficient scientific evidence  
            Output: {{ "SUPPORT": 0.0, "REFUTE": 1.0, "NOT_ENOUGH_INFO": 0.0 }}

            ---

            Example 2:  
            Statement: "Climate change is contributing to more frequent and intense heatwaves."  
            SUPPORT: Scientific evidence supports this statement  
            REFUTE: Scientific evidence contradicts this statement  
            NOT_ENOUGH_INFO: Insufficient scientific evidence  
            Output: {{ "SUPPORT": 0.8, "REFUTE": 0.15, "NOT_ENOUGH_INFO": 0.05 }}

            ---

            Example 3:  
            Statement: "A large number of people are worried about the rising sea levels."  
            SUPPORT: Scientific evidence supports this statement  
            REFUTE: Scientific evidence contradicts this statement  
            NOT_ENOUGH_INFO: Insufficient scientific evidence  
            Output: {{ "SUPPORT": 0.0, "REFUTE": 0.0, "NOT_ENOUGH_INFO": 1.0 }}

            ---

            Example 4:  
            Statement: "Some scientists believe solar activity is the primary cause of recent global warming."  
            SUPPORT: Scientific evidence supports this statement  
            REFUTE: Scientific evidence contradicts this statement  
            NOT_ENOUGH_INFO: Insufficient scientific evidence  
            Output: {{ "SUPPORT": 0.2, "REFUTE": 0.7, "NOT_ENOUGH_INFO": 0.1 }}

            ---

            Now analyze the following statement:

            Statement: "{text}"

            SUPPORT: Scientific evidence supports this statement  
            REFUTE: Scientific evidence contradicts this statement  
            NOT_ENOUGH_INFO: Insufficient scientific evidence

            Format: {{ "SUPPORT": 0.X, "REFUTE": 0.X, "NOT_ENOUGH_INFO": 0.X }}
            """

            
            response = self.gemini_model.generate_content(prompt)
            
            # Extract JSON more reliably
            import re
            text_response = response.text
            
            # Try multiple patterns to find JSON
            patterns = [
                r'\{[^{}]*"SUPPORT"[^{}]*\}',
                r'\{.*?"SUPPORT".*?\}',
                r'"SUPPORT":\s*[\d.]+.*?"REFUTE":\s*[\d.]+.*?"NOT_ENOUGH_INFO":\s*[\d.]+'
            ]
            
            result = None
            for pattern in patterns:
                match = re.search(pattern, text_response, re.DOTALL | re.IGNORECASE)
                if match:
                    try:
                        # Clean up the match and try to parse
                        json_str = match.group(0)
                        if not json_str.startswith('{'):
                            json_str = '{' + json_str + '}'
                        result = json.loads(json_str)
                        break
                    except:
                        continue
            
            if result and all(key in result for key in ["SUPPORT", "REFUTE", "NOT_ENOUGH_INFO"]):
                # Normalize scores
                total = sum(result[key] for key in ["SUPPORT", "REFUTE", "NOT_ENOUGH_INFO"])
                if total > 0:
                    for key in ["SUPPORT", "REFUTE", "NOT_ENOUGH_INFO"]:
                        result[key] = result[key] / total
                    
                    logger.info(f"Gemini: S:{result['SUPPORT']:.2f} R:{result['REFUTE']:.2f} N:{result['NOT_ENOUGH_INFO']:.2f}")
                    return result
            
            return self.get_default_scores("Gemini parse failed")
            
        except Exception as e:
            logger.error(f"Gemini prediction error: {e}")
            return self.get_default_scores("Gemini error")

    def get_model_prediction(self, text: str) -> Dict[str, Any]:
        """Get model prediction with proper shape handling"""
        try:
            if not self.trained_model or not self.tokenizer:
                return self.get_default_scores("Model unavailable")
            
           
            
            inputs = self.tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True, 
                padding=True,
                max_length=512
            )
            
            with torch.no_grad():
                outputs = self.trained_model(**inputs)
                logits = outputs.logits
                probs = torch.nn.functional.softmax(logits, dim=-1)
                
                # Handle different output shapes properly
                if probs.dim() > 1:
                    probs = probs.squeeze()
                
                probs_np = probs.cpu().numpy()
                
                # Ensure we have a 1D array
                if probs_np.ndim > 1:
                    probs_np = probs_np[0]
                
                logger.debug(f"Model output shape: {probs_np.shape}, values: {probs_np}")
                
                # Handle different numbers of output classes
                if len(probs_np) == 2:
                    # Binary classification - assume 0=REFUTE, 1=SUPPORT
                    result = {
                        "SUPPORT": float(probs_np[1]),
                        "REFUTE": float(probs_np[0]),
                        "NOT_ENOUGH_INFO": 0.1
                    }
                elif len(probs_np) == 3:
                    # Perfect - 3 classes as expected
                    result = {
                        "SUPPORT": float(probs_np[0]),
                        "REFUTE": float(probs_np[1]),
                        "NOT_ENOUGH_INFO": float(probs_np[2])
                    }
                elif len(probs_np) == 1:
                    # Single output - treat as confidence
                    conf = float(probs_np[0])
                    result = {
                        "SUPPORT": conf,
                        "REFUTE": 1.0 - conf,
                        "NOT_ENOUGH_INFO": 0.1
                    }
                else:
                    # Unexpected number of classes
                    logger.warning(f"Unexpected model output size: {len(probs_np)}")
                    return self.get_default_scores("Unexpected model output")
                
                # Normalize to ensure sum = 1.0
                total = sum(result.values())
                if total > 0:
                    for key in result:
                        result[key] = result[key] / total
                
                logger.info(f"BERT: S:{result['SUPPORT']:.2f} R:{result['REFUTE']:.2f} N:{result['NOT_ENOUGH_INFO']:.2f}")
                return result
            
        except Exception as e:
            logger.error(f"Model prediction error: {e}")
            return self.get_default_scores("Model error")

    def combine_predictions(self, gemini_pred: Dict, model_pred: Dict) -> Dict[str, Any]:
        """Combine predictions safely"""
        try:
            combined = {
                "SUPPORT": (gemini_pred["SUPPORT"] + model_pred["SUPPORT"]) / 2,
                "REFUTE": (gemini_pred["REFUTE"] + model_pred["REFUTE"]) / 2,
                "NOT_ENOUGH_INFO": (gemini_pred["NOT_ENOUGH_INFO"] + model_pred["NOT_ENOUGH_INFO"]) / 2
            }
            
            final_class = max(combined, key=combined.get)
            confidence = combined[final_class]
            
            logger.info(f"Ensemble: {final_class} ({confidence:.2%})")
            
            return {
                "prediction": final_class,
                "confidence": float(confidence),
                "scores": combined
            }
            
        except Exception as e:
            logger.error(f"Ensemble error: {e}")
            return {
                "prediction": "NOT_ENOUGH_INFO",
                "confidence": 0.34,
                "scores": self.get_default_scores("Ensemble error")
            }

    def generate_lime_explanation(self, text: str) -> List[Tuple[str, float]]:
        """Generate LIME explanation returning (word, score) tuples sorted by absolute score"""
        try:
            if not self.lime_explainer or not self.trained_model or not self.tokenizer:
                return [("lime", 0.0), ("unavailable", 0.0)]
            
            def predict_fn(texts):
                try:
                    if isinstance(texts, str):
                        texts = [texts]
                    
                    inputs = self.tokenizer(
                        list(texts), 
                        return_tensors="pt", 
                        padding=True, 
                        truncation=True,
                        max_length=512
                    )
                    
                    with torch.no_grad():
                        outputs = self.trained_model(**inputs)
                        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
                        probs_np = probs.cpu().numpy()
                        
                        if probs_np.shape[1] == 2:
                            expanded = np.zeros((probs_np.shape[0], 3))
                            expanded[:, 0] = probs_np[:, 1]
                            expanded[:, 1] = probs_np[:, 0]
                            expanded[:, 2] = 0.1
                            probs_np = expanded
                        elif probs_np.shape[1] == 1:
                            expanded = np.zeros((probs_np.shape[0], 3))
                            expanded[:, 0] = probs_np[:, 0]
                            expanded[:, 1] = 1.0 - probs_np[:, 0]
                            expanded[:, 2] = 0.1
                            probs_np = expanded
                        
                        row_sums = probs_np.sum(axis=1)
                        probs_np = probs_np / row_sums[:, np.newaxis]
                        
                        return probs_np
                        
                except Exception as e:
                    logger.error(f"LIME predict_fn error: {e}")
                    batch_size = len(texts) if isinstance(texts, list) else 1
                    return np.full((batch_size, 3), 0.33)
            
            explanation = self.lime_explainer.explain_instance(
                text, 
                predict_fn, 
                num_features=5
            )
            
            # Get list of (word, score) tuples
            words_scores = explanation.as_list()
            # Sort by absolute value of score descending
            words_scores.sort(key=lambda x: abs(x[1]), reverse=True)
            
            logger.info(f"🔍 LIME words with scores: {words_scores}")
            return words_scores
        
        except Exception as e:
            logger.error(f"LIME explanation error: {e}")
            return [("lime", 0.0), ("analysis_failed", 0.0)]

    def get_explanation(self, text: str, prediction: Dict) -> str:
       
        try:
            lime_words_with_scores = self.generate_lime_explanation(text)
            lime_words_str = ", ".join([f"{word} ({score:+.2f})" for word, score in lime_words_with_scores])
            
            if self.gemini_model:
                prompt = f"""
            The model classified the following statement as **{prediction['prediction']}** with {prediction['confidence']:.1f}% confidence.

            Statement: "{text}"

            Here are the key influential words and their influence scores on the model’s prediction:
            {lime_words_str}

            Note: Positive scores indicate words that support the predicted class, while negative scores indicate words that oppose it. The magnitude of the score reflects the strength of the influence.

            Using this information, briefly explain why the model likely made this prediction based on these words and their scores. 
            Ignore trivial or generic terms, and keep the explanation faithful to the sentence’s meaning.

            Then, write a short follow-up sentence that adds helpful scientific context or clarification related to the topic.
            This may include a real-world implication, fact, or environmental insight that helps the user understand the statement more fully.
            Keep everything concise and accurate.
            """
                response = self.gemini_model.generate_content(prompt)
                explanation = response.text.strip()
                logger.info(f"Explanation generated")
                return explanation
            
        except Exception as e:
            logger.error(f"Explanation error: {e}")
        
        return f"The AI ensemble classified this statement as {prediction['prediction']} with {prediction['confidence']:.1%} confidence. This assessment is based on patterns learned from climate science literature and training data."


# Basic endpoints
@app.get("/")
async def root():
    return {"message": "Climate Detection API", "status": "running"}

@app.get("/health")
async def health_check():
    global pipeline
    
    if not pipeline:
        return {"status": "initializing"}
    
    return {
        "status": "healthy",
        "components": {
            "gemini": pipeline.gemini_model is not None,
            "bert_model": pipeline.trained_model is not None,
            "lime": pipeline.lime_explainer is not None
        }
    }

@app.post("/detect", response_model=DetectionResult)
async def detect_misinformation(request: QueryRequest):
    """Main detection endpoint"""
    try:
        global pipeline
        
        if not pipeline:
            raise HTTPException(status_code=503, detail="Pipeline not ready")
        
        text = request.text.strip()
        
        if not text:
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        logger.info(f" Analyzing: {text}")
        
        # Domain check
        is_climate_related = pipeline.check_climate_domain(text)
        
        if not is_climate_related:
            return DetectionResult(
                is_climate_related=False,
                final_result={"message": "Not climate-related"}
            )
        
        # Get predictions
        gemini_pred = pipeline.get_gemini_prediction(text)
        model_pred = pipeline.get_model_prediction(text)
        
        # Combine
        final_result = pipeline.combine_predictions(gemini_pred, model_pred)
        
        # Explanation
        explanation = "for god sake"
        #explanation = pipeline.get_explanation(text, final_result)
        
        return DetectionResult(
            is_climate_related=True,
            gemini_prediction=gemini_pred,
            model_prediction=model_pred,
            final_result=final_result,
            explanation=explanation,
            confidence=final_result["confidence"]
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
        logger.info("Starting Climate Detection API...")
        pipeline = ClimateDetectionPipeline()
        logger.info("API ready!")
    except Exception as e:
        logger.error(f"Startup failed: {e}")

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting server on http://localhost:8000")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)