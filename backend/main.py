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
import google.generativeai as genai
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)

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
    gemini_prediction: Optional[Dict[str, Any]] = None
    model_prediction: Optional[Dict[str, Any]] = None
    final_result: Optional[Dict[str, Any]] = None
    explanation: Optional[str] = None
    confidence: Optional[float] = None
    word_attributions: Optional[List[Dict[str, Any]]] = None

# Global variables
pipeline = None
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_PATH = os.getenv("MODEL_PATH")

class ClimateDetectionPipeline:
    def __init__(self):
        self.gemini_model = None
        self.trained_model = None
        self.tokenizer = None
        self.transformer_explainer = None
        
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
    #loading the trained (fine-tuned) model
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

            # Test the model and check output shape which is 3 i.e `SUPPORT`, `REFUTE`, `NOT_ENOUGH_INFO`
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
            # Fallback to Roberta base model if loading fails
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
                # Keyword-based fallback incase Gemini is unavailable
                logger.info("Gemini model not available, using keyword-based domain check")
                climate_keywords = [
                    'climate', 'warming', 'carbon', 'temperature', 'greenhouse', 
                    'emissions', 'co2', 'fossil', 'renewable', 'environment',
                    'pollution', 'weather', 'arctic', 'ice', 'ocean', 'atmosphere'
                ]
                text_lower = text.lower()
                is_climate = any(keyword in text_lower for keyword in climate_keywords)
                logger.info(f"Keyword-based domain check: {'YES' if is_climate else 'NO'}")
                return is_climate
            #prompt to guide Gemini to check if the text is related to climate change
            prompt = f"""Determine whether the following statement is related to climate or climate change.This may include — but is not limited to — topics such as: global warming, greenhouse gases, CO₂ emissions, temperature rise, melting ice, sea level rise, fossil fuels, deforestation, extreme weather events, renewable energy, or other environmental changes linked to the Earth's climate.Respond with only YES or NO.Statement: {text}"""

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
            # Prompt to guide Gemini to evaluate the scientific credibility of the statement
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
        """Combine predictions safely
        Current approach is to average the scores from Gemini and the fine-tuned model.
        This can be adjusted based on performance and requirements.
        """
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

    def get_transformer_explanation(self, text: str, prediction: Dict) -> Tuple[List[Dict], str]:
        """Generate explanation using Transformer Interpret with subword merging and thresholding"""
        try:
            if not self.transformer_explainer:
                logger.info("Transformer Interpret: Not available - explainer not initialized")
                return [], "Transformer explanation not available."
            
            logger.info(f"Transformer Interpret: Starting analysis for prediction '{prediction['prediction']}'")
            
            # Get word attributions using integrated gradients
            raw_attributions = self.transformer_explainer(
                text,
                class_name=prediction['prediction'],  
                internal_batch_size=1
            )
            
            logger.info(f"🔍 Raw attributions received: {len(raw_attributions)} tokens")
            
            # Merge subwords and aggregate scores
            merged_tokens = []
            merged_scores = []
            current_token = ""
            current_score = 0.0
            
            for token, score in raw_attributions:
                if token.startswith("##"):
                    # This is a subword continuation
                    current_token += token[2:]  # Remove ##
                    current_score += score
                else:
                    # New word - save previous if exists
                    if current_token:
                        merged_tokens.append(current_token)
                        merged_scores.append(current_score)
                    current_token = token
                    current_score = score
            
            # Add the last token
            if current_token:
                merged_tokens.append(current_token)
                merged_scores.append(current_score)
            
            logger.info(f"🔍 After subword merging: {len(merged_tokens)} words")
            
            # Calculate dynamic threshold
            positive_scores = [s for s in merged_scores if s > 0]
            negative_scores = [s for s in merged_scores if s < 0]
            
            if positive_scores:
                mean_positive = np.mean(positive_scores)
                std_positive = np.std(positive_scores) if len(positive_scores) > 1 else 0
                positive_threshold = mean_positive + 0.5 * std_positive
            else:
                positive_threshold = 0
            
            if negative_scores:
                mean_negative = np.mean(negative_scores)
                std_negative = np.std(negative_scores) if len(negative_scores) > 1 else 0
                negative_threshold = mean_negative - 0.5 * std_negative  # More negative
            else:
                negative_threshold = 0
            
            logger.info(f"Threshold Analysis:")
            logger.info(f"   • Positive scores: {len(positive_scores)} words, mean={mean_positive:.3f}, std={std_positive:.3f}" if positive_scores else "   • No positive scores")
            logger.info(f"   • Positive threshold: {positive_threshold:.3f}")
            logger.info(f"   • Negative scores: {len(negative_scores)} words, mean={mean_negative:.3f}, std={std_negative:.3f}" if negative_scores else "   • No negative scores")
            logger.info(f"   • Negative threshold: {negative_threshold:.3f}")
            
            # Filter important tokens based on thresholds
            important_positive = [
                (token, score) for token, score in zip(merged_tokens, merged_scores) 
                if score > positive_threshold
            ]
            
            important_negative = [
                (token, score) for token, score in zip(merged_tokens, merged_scores) 
                if score < negative_threshold
            ]
            
            # Fallback: If no tokens pass threshold, show top tokens
            if not important_positive and positive_scores:
                sorted_positive = sorted(
                    [(token, score) for token, score in zip(merged_tokens, merged_scores) if score > 0],
                    key=lambda x: x[1], reverse=True
                )
                important_positive = sorted_positive[:2]
                logger.info(" No tokens passed positive threshold, using top 2 positive")
            
            if not important_negative and negative_scores:
                sorted_negative = sorted(
                    [(token, score) for token, score in zip(merged_tokens, merged_scores) if score < 0],
                    key=lambda x: x[1]  # Most negative first
                )
                important_negative = sorted_negative[:2]
                logger.info(" No tokens passed negative threshold, using top 2 negative")
            
            # Combine and create final attribution list
            all_important = important_positive + important_negative
            
            # Format for output
            attributions_list = []
            for token, score in all_important:
                attributions_list.append({
                    "word": token,
                    "attribution": float(score),
                    "importance": abs(float(score)),
                    "passed_threshold": True
                })
            
            # Add some non-threshold tokens for context (top remaining by absolute value)
            remaining_tokens = [
                (token, score) for token, score in zip(merged_tokens, merged_scores)
                if (token, score) not in all_important
            ]
            remaining_sorted = sorted(remaining_tokens, key=lambda x: abs(x[1]), reverse=True)
            
            for token, score in remaining_sorted[:3]:  # Add top 3 remaining
                attributions_list.append({
                    "word": token,
                    "attribution": float(score),
                    "importance": abs(float(score)),
                    "passed_threshold": False
                })
            
            # Sort final list by importance
            attributions_list.sort(key=lambda x: x["importance"], reverse=True)
            
            # Log the results
            threshold_passed = [attr for attr in attributions_list if attr["passed_threshold"]]
            threshold_failed = [attr for attr in attributions_list if not attr["passed_threshold"]]
            
            logger.info(f"Transformer Interpret Results:")
            logger.info(f"   • Total words analyzed: {len(merged_tokens)}")
            logger.info(f"   • Words passing threshold: {len(threshold_passed)}")
            logger.info(f"   • Method: Integrated Gradients with dynamic thresholding")
            logger.info(f"   • Target class: {prediction['prediction']}")
            
            if threshold_passed:
                logger.info(f"   • Important words (above threshold):")
                for attr in threshold_passed[:5]:
                    logger.info(f"     - {attr['word']}: {attr['attribution']:+.3f} ✓")
            
            if threshold_failed and threshold_passed:
                logger.info(f"   • Context words (below threshold):")
                for attr in threshold_failed[:3]:
                    logger.info(f"     - {attr['word']}: {attr['attribution']:+.3f}")
            
            if attributions_list:
                max_attr = max([attr["attribution"] for attr in attributions_list])
                min_attr = min([attr["attribution"] for attr in attributions_list])
                logger.info(f"   • Attribution range: {min_attr:.3f} to {max_attr:.3f}")
            
            # Generate human-readable explanation
            explanation_text = self._format_attribution_explanation_with_threshold(text,
                attributions_list, prediction, positive_threshold, negative_threshold
            )
            
            logger.info("Transformer Interpret: Analysis completed successfully")
            return attributions_list[:10], explanation_text
            
        except Exception as e:
            logger.error(f" Transformer Interpret ERROR: {e}")
            logger.error(f"   • Error type: {type(e).__name__}")
            return [], f"Error generating explanation: {str(e)}"

    def _format_attribution_explanation_with_threshold(self,text, attributions_list: List[Dict], prediction: Dict, pos_threshold: float, neg_threshold: float) -> str:
        """Use Gemini to create user-friendly explanation from Transformer Interpret results"""
        try:
            if not attributions_list:
                return "No significant word attributions found."
            
            # Get the most important words that passed threshold
            threshold_passed = [attr for attr in attributions_list if attr.get("passed_threshold", False)]
            
            # If no words passed threshold, use top words by importance
            if not threshold_passed:
                threshold_passed = sorted(attributions_list, key=lambda x: x["importance"], reverse=True)[:5]
            
            # Format words with scores for Gemini prompt
            transformer_words_str = ", ".join([
                f"{attr['word']} ({attr['attribution']:+.2f})" 
                for attr in threshold_passed[:8]  # Top 8 most important
            ])
            
            if self.gemini_model and transformer_words_str:
                prompt = f"""
The model classified the following statement as **{prediction['prediction']}** with {prediction['confidence']:.1f}% confidence.

Statement: "{text}"

Here are the key influential words and their influence scores on the model's prediction:
{transformer_words_str}

Note: Positive scores indicate words that support the predicted class, while negative scores indicate words that oppose it. The magnitude of the score reflects the strength of the influence.

Using this information, briefly explain why the model likely made this prediction based on these words and their scores. 
Ignore trivial or generic terms, and keep the explanation faithful to the sentence's meaning.

Then, write a short follow-up sentence that adds helpful scientific context or clarification related to the topic.
This may include a real-world implication, fact, or environmental insight that helps the user understand the statement more fully.
Keep everything concise and accurate.
"""
                
                response = self.gemini_model.generate_content(prompt)
                explanation = response.text.strip()
                logger.info("User-friendly explanation generated from Transformer Interpret results")
                return explanation
            
        except Exception as e:
            logger.error(f"Explanation generation error: {e}")
        
        # Fallback explanation
        pred_class = prediction['prediction']
        confidence = prediction['confidence']
        return f"The AI ensemble classified this statement as {pred_class} with {confidence:.1%} confidence. This assessment is based on patterns learned from climate science literature and training data."

    def get_explanation(self, text: str, prediction: Dict) -> Tuple[str, List[Dict]]:
        """Get comprehensive explanation combining Transformer Interpret and Gemini"""
        # Add original text to prediction for context
        prediction['original_text'] = text
        
        # Get Transformer Interpret explanation
        word_attributions, transformer_explanation = self.get_transformer_explanation(text, prediction)
        
        return transformer_explanation, word_attributions


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
            "transformer_interpret": pipeline.transformer_explainer is not None,
            "transformer_interpret_available": TRANSFORMER_INTERPRET_AVAILABLE
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
        
        logger.info(f"Analyzing: {text}")
        
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
        
        # Combine predictions
        final_result = pipeline.combine_predictions(gemini_pred, model_pred)
        
        # Get explanations and word attributions
        explanation, word_attributions = pipeline.get_explanation(text, final_result)
        
        return DetectionResult(
            is_climate_related=True,
            gemini_prediction=gemini_pred,
            model_prediction=model_pred,
            final_result=final_result,
            explanation=explanation,
            confidence=final_result["confidence"],
            word_attributions=word_attributions
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