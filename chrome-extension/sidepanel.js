

const API_BASE_URL = 'https://cmi-with-xai.fly.dev'


// DOM elements
const statementInput = document.getElementById('statementInput');
const analyzeBtn = document.getElementById('analyzeBtn');
const clearBtn = document.getElementById('clearBtn');
const selectTextBtn = document.getElementById('selectTextBtn');
const statusIndicator = document.getElementById('statusIndicator');
const loading = document.getElementById('loading');
const results = document.getElementById('results');
const incompleteStatement = document.getElementById('incompleteStatement');
const outOfDomain = document.getElementById('outOfDomain');
const error = document.getElementById('error');
const errorMessage = document.getElementById('errorMessage');

// Result elements
const resultClassification = document.getElementById('resultClassification');
const overallConfidence = document.getElementById('overallConfidence');
const supportFill = document.getElementById('supportFill');
const refuteFill = document.getElementById('refuteFill');
const explanationContent = document.getElementById('explanationContent');
const guidanceContent = document.getElementById('guidanceContent');

// State
let isSelectionMode = false;

// Debug logging function
function debugLog(message, data = null) {
  console.log(`[SIDEPANEL] ${message}`, data);
}

// Event listeners
analyzeBtn.addEventListener('click', analyzeStatement);
clearBtn.addEventListener('click', clearResults);
selectTextBtn.addEventListener('click', toggleTextSelection);

statementInput.addEventListener('keypress', (e) => {
  if (e.key === 'Enter' && e.ctrlKey) {
    analyzeStatement();
  }
});

// Listen for messages from background script or content script
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  debugLog('Message received:', request);
  
  if (request.action === 'textSelectedForAnalysis') {
    // Side panel is open and received selected text directly
    debugLog('Received text directly for analysis:', request.text.substring(0, 100));
    
    // Fill textarea immediately
    statementInput.value = request.text;
    
    // Save to storage
    chrome.storage.local.set({ lastStatement: request.text });
    
    // Update status
    statusIndicator.textContent = 'Analyzing selected text...';
    statusIndicator.className = 'status-indicator status-selecting';
    
    // Start analysis immediately
    setTimeout(() => {
      debugLog('Analyzing text received via message');
      analyzeStatement();
    }, 500);
    
    sendResponse({ success: true });
  } else if (request.action === 'prepareForTextSelection') {
    debugLog('Preparing for text selection');
    updateStatusForSelection();
    sendResponse({ success: true });
  }
  
  return true; // Keep message channel open
});

// Load saved statement and check for pending analysis
document.addEventListener('DOMContentLoaded', () => {
  debugLog('DOM Content Loaded');
  
  chrome.storage.local.get(['lastStatement', 'pendingAnalysis', 'autoAnalyze'], (result) => {
    debugLog('Storage data loaded:', result);
    
    if (result.pendingAnalysis && result.autoAnalyze) {
      // Only auto-analyze if this is fresh (within last 10 seconds)
      const now = Date.now();
      const timestamp = result.timestamp || 0;
      const timeDiff = now - timestamp;
      
      debugLog('Time difference:', timeDiff, 'ms');
      
      if (timeDiff < 10000) { // 10 seconds
        // Fresh selection - auto-analyze
        debugLog('Auto-loading FRESH selected text:', result.pendingAnalysis.substring(0, 100));
        statementInput.value = result.pendingAnalysis;
        
        // Clear the auto-analyze flag immediately
        chrome.storage.local.remove(['pendingAnalysis', 'autoAnalyze']);
        
        // Update status
        statusIndicator.textContent = 'Analyzing selected text...';
        statusIndicator.className = 'status-indicator status-selecting';
        
        // Auto-analyze
        setTimeout(() => {
          debugLog('Auto-analyzing FRESH selected text');
          analyzeStatement();
        }, 1000);
      } else {
        // Old selection - just load text but don't auto-analyze
        debugLog('OLD selection detected, just loading text without auto-analysis');
        statementInput.value = result.pendingAnalysis;
        
        // Clear the old data
        chrome.storage.local.remove(['pendingAnalysis', 'autoAnalyze']);
      }
      
    } else if (result.lastStatement) {
      // Load previous statement but don't auto-analyze
      statementInput.value = result.lastStatement;
      debugLog('Loaded saved statement (no auto-analysis):', result.lastStatement.substring(0, 100));
    } else {
      debugLog('No stored data found');
    }
  });
});

// Save statement when typing
statementInput.addEventListener('input', () => {
  chrome.storage.local.set({ lastStatement: statementInput.value });
});

async function toggleTextSelection() {
  try {
    // Prevent rapid toggling
    if (selectTextBtn.disabled) {
      debugLog('Button disabled, ignoring click');
      return;
    }
    
    selectTextBtn.disabled = true;
    debugLog('Text selection button clicked in side panel');
    
    // Get current active tab
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    
    if (!tab) {
      throw new Error('No active tab found');
    }
    
    debugLog('Active tab found:', tab.url);
    
    // Check if we can access this tab
    if (tab.url.startsWith('chrome://') || tab.url.startsWith('chrome-extension://') || tab.url.startsWith('edge://')) {
      throw new Error('Text selection not available on browser pages. Try on a regular website.');
    }
    
    // Always enter selection mode when button is clicked from side panel
    selectTextBtn.textContent = 'Selection Active';
    selectTextBtn.classList.add('active');
    selectTextBtn.style.background = '#4CAF50';
    statusIndicator.textContent = '📖 Select text from the webpage';
    statusIndicator.className = 'status-indicator status-selecting';
    
    debugLog('Activating text selection on tab:', tab.id);
    
    try {
      // First try to send message to existing content script
      await chrome.tabs.sendMessage(tab.id, { action: 'enterSelectionMode' });
      debugLog('Message sent to existing content script');
    } catch (msgError) {
      debugLog('No existing content script, injecting directly');
      
      // If content script doesn't exist, inject selection mode directly
      await chrome.scripting.executeScript({
        target: { tabId: tab.id },
        func: () => {
          console.log('[INJECTED] Direct selection mode activation from side panel');
          
          // Create selection overlay
          const overlay = document.createElement('div');
          overlay.id = 'climate-selection-overlay-direct';
          overlay.innerHTML = `
            <div style="
              position: fixed;
              top: 20px;
              right: 20px;
              background: #4CAF50;
              color: white;
              padding: 15px 20px;
              border-radius: 10px;
              font-family: Arial, sans-serif;
              font-size: 14px;
              font-weight: bold;
              z-index: 999999;
              box-shadow: 0 6px 20px rgba(0,0,0,0.3);
              border: 2px solid rgba(255,255,255,0.3);
              cursor: pointer;
            ">
               SELECTION MODE ACTIVE<br>
              <span style="font-size: 12px; opacity: 0.9;">Select text • ESC to exit • Click here to exit</span>
            </div>
          `;
          
          document.body.appendChild(overlay);
          document.body.style.cursor = 'crosshair';
          
          // Add selection handler
          function handleSelection(event) {
            setTimeout(() => {
              const selectedText = window.getSelection().toString().trim();
              if (selectedText && selectedText.length >= 10) {
                // Send directly to background
                chrome.runtime.sendMessage({
                  action: 'analyzeSelectedText',
                  text: selectedText,
                  timestamp: Date.now()
                });
                
                // Clean up
                document.body.style.cursor = '';
                if (overlay.parentNode) overlay.remove();
                document.removeEventListener('mouseup', handleSelection);
                window.getSelection().removeAllRanges();
                
                // Show success message
                const success = document.createElement('div');
                success.style.cssText = `
                  position: fixed; top: 50%; left: 50%; transform: translate(-50%, -50%);
                  background: #4CAF50; color: white; padding: 20px; border-radius: 12px;
                  font-family: Arial; font-size: 16px; font-weight: bold; z-index: 999999;
                  box-shadow: 0 8px 24px rgba(0,0,0,0.3); text-align: center;
                `;
                success.innerHTML = 'Text sent to analysis panel!';
                document.body.appendChild(success);
                setTimeout(() => success.remove(), 2000);
              }
            }, 100);
          }
          
          // Add exit handler
          function handleExit(event) {
            if (event.key === 'Escape') {
              document.body.style.cursor = '';
              if (overlay.parentNode) overlay.remove();
              document.removeEventListener('mouseup', handleSelection);
              document.removeEventListener('keydown', handleExit);
            }
          }
          
          // Exit on overlay click
          overlay.addEventListener('click', () => {
            document.body.style.cursor = '';
            overlay.remove();
            document.removeEventListener('mouseup', handleSelection);
            document.removeEventListener('keydown', handleExit);
          });
          
          document.addEventListener('mouseup', handleSelection);
          document.addEventListener('keydown', handleExit);
          
          return 'Direct selection mode activated successfully';
        }
      });
    }
    
    debugLog('Text selection activated successfully');
    
    // Reset button after 3 seconds
    setTimeout(() => {
      selectTextBtn.textContent = 'Select from Web Page';
      selectTextBtn.classList.remove('active');
      selectTextBtn.style.background = '';
      statusIndicator.textContent = 'Ready to analyze climate statements';
      statusIndicator.className = 'status-indicator status-ready';
    }, 3000);
    
  } catch (error) {
    debugLog('Error activating text selection:', error);
    showError('Failed to activate text selection: ' + error.message);
    
    // Reset button state on error
    selectTextBtn.textContent = 'Select from WebPage';
    selectTextBtn.classList.remove('active');
    selectTextBtn.style.background = '';
    statusIndicator.textContent = 'Selection failed - ' + error.message;
    statusIndicator.className = 'status-indicator status-ready';
  } finally {
    // Re-enable button after delay
    setTimeout(() => {
      selectTextBtn.disabled = false;
    }, 1000);
  }
}

function updateStatusForSelection() {
  statusIndicator.textContent = 'Text selection mode ready';
  statusIndicator.className = 'status-indicator status-selecting';
}

function handleSelectedText(text) {
  debugLog('Handling selected text:', text ? text.substring(0, 100) + '...' : 'NO TEXT RECEIVED');
  
  if (!text || text.trim().length < 5) {
    showError('Selected text is too short for analysis');
    return;
  }
  
  // Reset selection mode UI
  isSelectionMode = false;
  selectTextBtn.textContent = 'Select from Web Page';
  selectTextBtn.classList.remove('active');
  statusIndicator.textContent = 'Text selected! Analyzing...';
  statusIndicator.className = 'status-indicator status-ready';
  
  // Fill the textarea
  statementInput.value = text;
  
  // Save to storage
  chrome.storage.local.set({ lastStatement: text });
  
  // Auto-analyze immediately
  setTimeout(() => {
    debugLog('Auto-analyzing selected text');
    analyzeStatement();
  }, 1000);
}

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
  
  // Handle different response statuses
  switch (data.status) {
    case 'out_of_domain':
      debugLog('Statement not climate-related');
      outOfDomain.classList.remove('hidden');
      return;
      
    case 'incomplete_statement':
      debugLog('Statement incomplete');
      showIncompleteStatement(data.completeness_guidance);
      return;
      
    case 'complete_analysis':
      debugLog('Complete analysis received');
      showCompleteAnalysis(data);
      return;
      
    default:
      debugLog('Unknown status:', data.status);
      showError('Unexpected response format');
      return;
  }
}

function showIncompleteStatement(guidance) {
  debugLog('Showing incomplete statement guidance:', guidance);
  
  if (guidance) {
    guidanceContent.textContent = guidance;
  } else {
    guidanceContent.textContent = "Please provide more specific details about the conditions, location, timeframe, or context for accurate analysis.";
  }
  
  incompleteStatement.classList.remove('hidden');
}

function showCompleteAnalysis(data) {
  debugLog('Showing complete analysis:', data);
  
  if (!data.final_result) {
    debugLog('ERROR: No final_result in response data');
    showError('Invalid response format - missing final_result');
    return;
  }
  
  const finalResult = data.final_result;
  const prediction = finalResult.prediction;
  const confidence = finalResult.confidence;
  
  debugLog('Extracted values:', { prediction, confidence });
  
  // Update main classification with new messaging
  updateClassification(prediction, confidence);
  
  // Update confidence visualizations for binary classification
  updateBinaryConfidenceVisuals(finalResult);
  
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
  
  let classificationText, classificationClass;
  
  if (prediction === 'SUPPORT') {
    classificationText = 'Statement is TRUTHFUL';
    classificationClass = 'truthful';
  } else { // REFUTE
    classificationText = 'Statement is MISINFORMATION';
    classificationClass = 'misinformation';
  }
  
  debugLog('Classification info:', { classificationText, classificationClass });
  
  resultClassification.textContent = classificationText;
  resultClassification.className = `result-classification ${classificationClass}`;
  
  overallConfidence.textContent = `${(confidence * 100).toFixed(1)}% Confidence`;
  
  debugLog('Classification UI updated');
}

function updateBinaryConfidenceVisuals(finalResult) {
  debugLog('updateBinaryConfidenceVisuals called with finalResult:', finalResult);
  
  // For binary classification, we have support_score and refute_score
  const supportScore = finalResult.support_score || 0;
  const refuteScore = finalResult.refute_score || 0;
  
  debugLog('Binary scores:', { supportScore, refuteScore });
  
  // Update progress bars with staggered animation
  setTimeout(() => updateProgressBar(supportFill, supportScore), 200);
  setTimeout(() => updateProgressBar(refuteFill, refuteScore), 400);
}

function updateProgressBar(element, value) {
  const percentage = Math.round(value * 100);
  element.style.width = `${percentage}%`;

  // Adjust position based on percentage
  if (percentage < 15) {
    element.style.justifyContent = 'flex-start';
    element.style.paddingLeft = '5px';
    element.style.paddingRight = '0px';
    element.style.color = '#333';
    element.style.textShadow = 'none';
  } else {
    element.style.justifyContent = 'flex-end';
    element.style.paddingLeft = '0px';
    element.style.paddingRight = '10px';
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
      The AI system has analyzed this statement using climate science evidence and machine learning models.
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
  incompleteStatement.classList.add('hidden');
  outOfDomain.classList.add('hidden');
  error.classList.add('hidden');
}

function setLoading(isLoading) {
  debugLog('setLoading called', { isLoading });

  if (isLoading) {
    loading.classList.remove('hidden');
    analyzeBtn.disabled = true;
    analyzeBtn.textContent = 'Analyzing...';

    // Disable "Select from Page" button with inline styles
    selectTextBtn.disabled = true;
    selectTextBtn.style.opacity = '0.6';
    selectTextBtn.style.cursor = 'not-allowed';
  } else {
    loading.classList.add('hidden');
    analyzeBtn.disabled = false;
    analyzeBtn.textContent = 'Analyze Statement';

    // Re-enable "Select from Page" button with inline styles reset
    selectTextBtn.disabled = false;
    selectTextBtn.style.opacity = '';
    selectTextBtn.style.cursor = '';
  }
}

function clearResults() {
  debugLog('clearResults called');
  
  statementInput.value = '';
  hideAllResults();
  chrome.storage.local.remove(['lastStatement']);
  
  // Reset progress bars (only 2 now for binary classification)
  [supportFill, refuteFill].forEach(element => {
    element.style.width = '0%';
    element.textContent = '0%';
  });
  
  // Reset status
  statusIndicator.textContent = 'Ready to analyze climate statements';
  statusIndicator.className = 'status-indicator status-ready';
  
  statementInput.focus();
}

window.addEventListener('error', (event) => {
  debugLog('Global error caught:', event.error);
});