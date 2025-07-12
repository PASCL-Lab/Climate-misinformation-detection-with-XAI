const API_BASE_URL = 'http://localhost:8000';

// DOM elements
const statementInput = document.getElementById('statementInput');
const analyzeBtn = document.getElementById('analyzeBtn');
const clearBtn = document.getElementById('clearBtn');
const loading = document.getElementById('loading');
const results = document.getElementById('results');
const outOfDomain = document.getElementById('outOfDomain');
const error = document.getElementById('error');
const errorMessage = document.getElementById('errorMessage');

// Result elements
const resultClassification = document.getElementById('resultClassification');
const overallConfidence = document.getElementById('overallConfidence');
const supportFill = document.getElementById('supportFill');
const refuteFill = document.getElementById('refuteFill');
const uncertainFill = document.getElementById('uncertainFill');
const explanationContent = document.getElementById('explanationContent');

// Debug logging function
function debugLog(message, data = null) {
  console.log(`[DEBUG] ${message}`, data);
}

// Event listeners
analyzeBtn.addEventListener('click', analyzeStatement);
clearBtn.addEventListener('click', clearResults);

statementInput.addEventListener('keypress', (e) => {
  if (e.key === 'Enter' && e.ctrlKey) {
    analyzeStatement();
  }
});

// Load saved statement on popup open
document.addEventListener('DOMContentLoaded', () => {
  debugLog('DOM Content Loaded');
  
  chrome.storage.local.get(['lastStatement'], (result) => {
    if (result.lastStatement) {
      statementInput.value = result.lastStatement;
      debugLog('Loaded saved statement', result.lastStatement);
    }
  });
});

// Save statement when typing
statementInput.addEventListener('input', () => {
  chrome.storage.local.set({ lastStatement: statementInput.value });
});

async function analyzeStatement() {
  const statement = statementInput.value.trim();
  
  debugLog('Analysis started', { statement });
  
  if (!statement) {
    showError('Please enter a statement to analyze.');
    return;
  }

  if (statement.length < 5) {
    showError('Please enter a longer statement for accurate analysis.');
    return;
  }

  setLoading(true);
  hideAllResults();
  
  try {
    debugLog('Making API request to:', `${API_BASE_URL}/detect`);
    
    const response = await fetch(`${API_BASE_URL}/detect`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ text: statement })
    });
    
    debugLog('API response received', { 
      status: response.status, 
      statusText: response.statusText,
      ok: response.ok 
    });
    
    if (!response.ok) {
      if (response.status === 404) {
        throw new Error('Backend server not found. Make sure your API is running on localhost:8000');
      } else if (response.status >= 500) {
        throw new Error('Server error. Please try again in a moment.');
      } else {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
    }
    
    const data = await response.json();
    debugLog('API response data:', data);
    
    displayResults(data);
    
  } catch (err) {
    debugLog('API Error:', err);
    console.error('Analysis error:', err);
    if (err.message.includes('fetch')) {
      showError('Cannot connect to the analysis server. Make sure your backend is running.');
    } else {
      showError(err.message || 'Failed to analyze statement. Please try again.');
    }
  } finally {
    setLoading(false);
  }
}

function displayResults(data) {
  debugLog('displayResults called with data:', data);
  
  hideAllResults();
  
  if (!data.is_climate_related) {
    debugLog('Statement not climate-related');
    outOfDomain.classList.remove('hidden');
    return;
  }
  
  // Check if we have final_result
  if (!data.final_result) {
    debugLog('ERROR: No final_result in response data');
    showError('Invalid response format - missing final_result');
    return;
  }
  
  const finalResult = data.final_result;
  debugLog('Final result extracted:', finalResult);
  
  const prediction = finalResult.prediction;
  const confidence = finalResult.confidence;
  const scores = finalResult.scores;
  debugLog('Scores extracted:', scores);
  
  debugLog('Extracted values:', { prediction, confidence, scores });
  
  if (!scores) {
    debugLog('ERROR: No scores in final_result');
    showError('Invalid response format - missing scores');
    return;
  }
  
  // Update main classification
  updateClassification(prediction, confidence);
  results.classList.remove('hidden');
  // Update confidence visualizations
  updateConfidenceVisuals(scores);
  
  // Update explanation
  updateExplanation(data.explanation);
  
  // Show results with animation
  setTimeout(() => {
    results.classList.remove('hidden');
    debugLog('Results displayed');
  }, 100);
}

function updateClassification(prediction, confidence) {
  debugLog('updateClassification called', { prediction, confidence });
  
  const classifications = {
    'SUPPORT': {
      text: 'Evidence Supports This Statement',
      class: 'support'
    },
    'REFUTE': {
      text: 'Evidence Refutes This Statement', 
      class: 'refute'
    },
    'NOT_ENOUGH_INFO': {
      text: 'Insufficient Evidence Available',
      class: 'not_enough_info'
    }
  };
  
  const classInfo = classifications[prediction] || {
    text: 'Analysis Complete',
    class: 'not_enough_info'
  };
  
  debugLog('Classification info:', classInfo);
  
  resultClassification.textContent = classInfo.text;
  resultClassification.className = `result-classification ${classInfo.class}`;
  
  overallConfidence.textContent = `${(confidence * 100).toFixed(1)}% Confidence`;
  
  debugLog('Classification UI updated');
}

function updateConfidenceVisuals(scores) {
  debugLog('updateConfidenceVisuals called with scores:', scores);
  
  // Validate scores object
  const requiredKeys = ['SUPPORT', 'REFUTE', 'NOT_ENOUGH_INFO'];
  for (const key of requiredKeys) {
    if (!(key in scores)) {
      debugLog(`ERROR: Missing score key: ${key}`);
      showError(`Invalid scores format - missing ${key}`);
      return;
    }
  }
  
  debugLog('Scores validation passed');
  
  // Update progress bars with staggered animation
  setTimeout(() => updateProgressBar(supportFill, scores.SUPPORT), 200);
  setTimeout(() => updateProgressBar(refuteFill, scores.REFUTE), 400);
  setTimeout(() => updateProgressBar(uncertainFill, scores.NOT_ENOUGH_INFO), 600);
}

function updateProgressBar(element, value) {
  const percentage = Math.round(value * 100);
  element.style.width = `${percentage}%`;

  // Adjust position or fallback
  if (percentage < 15) {
    element.style.justifyContent = 'flex-start';
    element.style.paddingLeft = '5px';
    element.style.color = '#333'; // dark for visibility
    element.style.textShadow = 'none';
  } else {
    element.style.justifyContent = 'flex-end';
    element.style.paddingLeft = '0px';
    element.style.color = 'white';
    element.style.textShadow = '0 1px 2px rgba(0,0,0,0.2)';
  }

  animateNumber(element, 0, percentage, 800);
}


function animateNumber(element, start, end, duration) {
  const startTime = performance.now();
  
  function update(currentTime) {
    const elapsed = currentTime - startTime;
    const progress = Math.min(elapsed / duration, 1);
    
    // Easing function for smooth animation
    const easeOut = 1 - Math.pow(1 - progress, 3);
    const current = Math.round(start + (end - start) * easeOut);
    
    element.textContent = `${current}%`;
    
    if (progress < 1) {
      requestAnimationFrame(update);
    }
  }
  
  requestAnimationFrame(update);
}

function updateExplanation(explanation) {
  debugLog('updateExplanation called', { explanation });
  
  if (explanation && explanation.trim()) {
    explanationContent.textContent = explanation;
  } else {
    explanationContent.innerHTML = `
      <em>No detailed explanation available for this analysis.</em>
      <br><br>
      The AI ensemble has provided confidence scores based on training data and analysis patterns.
    `;
  }
  
  debugLog('Explanation updated');
}

function showError(message) {
  debugLog('showError called', { message });
  hideAllResults();
  errorMessage.textContent = message;
  error.classList.remove('hidden');
}

function hideAllResults() {
  results.classList.add('hidden');
  outOfDomain.classList.add('hidden');
  error.classList.add('hidden');
}

function setLoading(isLoading) {
  debugLog('setLoading called', { isLoading });
  
  if (isLoading) {
    loading.classList.remove('hidden');
    analyzeBtn.disabled = true;
    analyzeBtn.textContent = 'Analyzing...';
  } else {
    loading.classList.add('hidden');
    analyzeBtn.disabled = false;
    analyzeBtn.textContent = 'Analyze Statement';
  }
}

function clearResults() {
  debugLog('clearResults called');
  
  statementInput.value = '';
  hideAllResults();
  chrome.storage.local.remove(['lastStatement']);
  
  // Reset progress bars
  [supportFill, refuteFill, uncertainFill].forEach(element => {
    element.style.width = '0%';
    element.textContent = '0%';
  });
  
  statementInput.focus();
}

// Add error handling
window.addEventListener('error', (event) => {
  debugLog('Global error caught:', event.error);
});