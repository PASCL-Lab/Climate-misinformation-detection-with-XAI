# Climate Misinformation Detector


The Climate Misinformation Detector is a browser extension that  analyze climate-related statements and determine their scientific accuracy. It features both manual text input and smart webpage text selection for  fact-checking.

## Features

### Dual Analysis Interface
- **Quick Popup Analysis** - Fast fact-checking directly in the popup
- **Persistent Side Panel** - Full-featured analysis that stays open while you browsing webpages

### Smart Text Selection
- Select text directly from any webpage with visual feedback
- Crosshair cursor and confirmation dialogs for precise selection
- Automatic text transfer to analysis panel

### AI-Powered Analysis
- **Ensemble  Model** combining Google Gemini and fine-tuned Climate-BERT
- **Binary classification**: True and False
- **Confidence scoring** with visual progress bars
- **Explanations** showing which words influenced the decision of the model ensuring transparency and trust for users.



## Installation

### Prerequisites

1. **Python 3.8+** with the following packages:
   ```bash
   pip install fastapi uvicorn python-dotenv torch transformers google-generativeai transformers-interpret
   ```

2. **Google Chrome** or Chromium-based browser

3. **API Keys**:
   - Google Gemini API key


### Backend Setup - Locally

1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd climate-misinformation-detector
   ```

2. **Set up environment variables**:
   Create a `.env` file:
   ```env
   GEMINI_API_KEY=your-gemini-api-key-here
   ```

3. **Start the backend server**:
   ```bash
   python main.py
   ```
   The server will start on `http://localhost:8000`

### Extension Installation

1. **Open Chrome** and navigate to `chrome://extensions/`

2. **Enable Developer Mode** (toggle in top-right corner)

3. **Load the extension**:
   - Click "Load unpacked"
   - Select the `chrome-extension` folder from this project

4. **Verify installation**:
   - Extension icon should appear in browser toolbar
   - Click icon to test popup functionality



## Usage Guide

### Method 1: Quick Analysis (Popup)

1. **Click the extension icon** in your browser toolbar
2. **Type or paste text** in the textarea
3. **Click "Quick Analyze"** 
4. **View results** directly in the popup with confidence scores and explanations

### Method 2: Webpage Text Selection

1. **Click extension icon** → **"Select from Page"**
2. **Side panel opens** automatically
3. **In side panel, click "Select from Page"** 
4. **Select text** on any webpage (crosshair cursor appears)
5. **Click "Analyze"** in confirmation dialog
6. **Results appear** automatically in side panel



## Models & Technology

### Ensemble Architecture

The system combines ensemble approach for analysis:

1. **Google Gemini 2.5 Pro**
   - Large language model for nuanced understanding
   - Evaluates scientific credibility against authoritative sources
   - Provides reasoning and context

2. **Fine-tuned climate-bert Model**
   - Specialized for climate misinformation detection
   - Trained on climate science literature
   

3. **Transformer Interpret Explainability**
   - Uses Integrated Gradients for word attribution analysis
   - Shows which words influenced the decision with numerical scores
   - Dynamic thresholding to identify most important terms
   - Provides transparency in AI reasoning

### Fine-tuned Model Details

#### Model Architecture
- **Base Model**: climate-bert
- **Training Framework**: Pytorch

#### Training Data for Finetuning
- **Dataset Size**: After preprocessing, our final dataset comprises 4,075 samples with the following distribution:
 • True: 2,149 samples (52.7%)
 • False: 1,926 samples (47.3%)
- **Data Sources**: CLIMATE-FEVER,Social Media platforms and LLM generated.
- **Labeling Methodology**: True-> Not misinformation False -> misinformation
- **Data Preprocessing**:  Conflict resolution strategies were applied for text received with conflicting labels and similarity-based detection was used to detect and manage exact and near-duplicate content.Lastly, standardized text preprocessing techniques were applied to ensure consistency. This included URL removal,hashtag normalization, mention removal, whitespace normalization case standardization and noise reduction.

#### Model Performance
- **Accuracy**: 0.892
- **Macro F1-Score**: 0.891


### Classification System

**True** (Green)
- Scientific evidence supports the statement
- Backed by peer-reviewed research and consensus

**False** (Red)  
- Scientific evidence contradicts the statement
- Conflicts with established climate science


## Configuration

### Backend Configuration

Edit `main.py` to customize:

```python
# API Configuration
API_BASE_URL = "http://localhost:8000"  # Change port if needed

# Model Configuration  
MODEL_PATH = "path/to/your/model"       # Custom model location
GEMINI_API_KEY = "your-api-key"         # Gemini API credentials

# Analysis Parameters
MIN_CONFIDENCE_THRESHOLD = 0.6          # Minimum confidence for decisions
MAX_TEXT_LENGTH = 1000                  # Maximum characters to analyze
```

### Extension Configuration

Edit `manifest.json` for permissions and settings:

```json
{
  "host_permissions": [
    "http://localhost:8000/*",          // Backend server
    "https://your-domain.com/*"         // Production server
  ],
  "content_scripts": [{
    "matches": ["<all_urls>"],          // Websites where text selection works
    "run_at": "document_end"
  }]
}
```




**Quick Start**: Install dependencies → Start `main.py` → Load extension → Start fact-checking! 

**Goal**: Combat climate misinformation with transparent, AI-powered analysis that helps users make informed decisions about climate science claims.It is for educational purposes only.