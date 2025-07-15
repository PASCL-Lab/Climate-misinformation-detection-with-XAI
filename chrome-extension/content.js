// Content script for text selection functionality
// Prevent multiple injections
(() => {
  if (window.hasRunClimateContentScript) {
    console.log('[CONTENT] Script already running, skipping re-injection.');
    return;
  }
  window.hasRunClimateContentScript = true;

  let isSelectionMode = false;
  let selectionOverlay = null;

  console.log("[CONTENT] Climate content script loaded");

  // Listen for messages from popup/sidepanel and window messages
  chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    console.log('[CONTENT] Message received:', request);
    
    if (request.action === 'toggleSelectionMode') {
      toggleSelectionMode();
      sendResponse({ success: true });
    } else if (request.action === 'enterSelectionMode') {
      enterSelectionMode();
      sendResponse({ success: true });
    } else if (request.action === 'exitSelectionMode') {
      exitSelectionMode();
      sendResponse({ success: true });
    } else if (request.action === 'showOpenSidePanelPrompt') {
      showOpenSidePanelPrompt(request.text);
      sendResponse({ success: true });
    }
  });

  // Listen for window messages (from injected scripts)
  window.addEventListener('message', (event) => {
    if (event.source !== window) return;
    if (event.data.source !== 'climate_extension') return;
    
    console.log('[CONTENT] Window message received:', event.data);
    
    if (event.data.action === 'ENTER_SELECTION_MODE') {
      enterSelectionMode();
    }
  });

  function toggleSelectionMode() {
    if (isSelectionMode) {
      exitSelectionMode();
    } else {
      enterSelectionMode();
    }
  }

  function enterSelectionMode() {
    if (isSelectionMode) return; // Already in selection mode
    
    isSelectionMode = true;
    console.log('[CONTENT] Entering selection mode');
    
    // Create overlay to indicate selection mode
    createSelectionOverlay();
    
    // Add event listeners
    document.addEventListener('mouseup', handleTextSelection, true);
    document.addEventListener('keydown', handleEscapeKey, true);
    
    // Change cursor to indicate selection mode
    document.body.style.cursor = 'crosshair';
    
    // Add visual indicator class to body
    document.body.classList.add('climate-selection-mode');
    
    // Add CSS for selection mode
    if (!document.getElementById('climate-selection-styles')) {
      const styles = document.createElement('style');
      styles.id = 'climate-selection-styles';
      styles.textContent = `
        .climate-selection-mode * {
          cursor: crosshair !important;
        }
      `;
      document.head.appendChild(styles);
    }
  }

  function exitSelectionMode() {
    if (!isSelectionMode) return; // Not in selection mode
    
    isSelectionMode = false;
    console.log('[CONTENT] Exiting selection mode');
    
    // Remove overlay
    removeSelectionOverlay();
    
    // Remove event listeners
    document.removeEventListener('mouseup', handleTextSelection, true);
    document.removeEventListener('keydown', handleEscapeKey, true);
    
    // Reset cursor
    document.body.style.cursor = '';
    
    // Remove visual indicator class
    document.body.classList.remove('climate-selection-mode');
    
    // Remove selection styles
    const styles = document.getElementById('climate-selection-styles');
    if (styles) styles.remove();
  }

  function createSelectionOverlay() {
    removeSelectionOverlay(); // Remove any existing overlay
    
    selectionOverlay = document.createElement('div');
    selectionOverlay.id = 'climate-detector-overlay';
    selectionOverlay.innerHTML = `
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
        animation: slideInRight 0.3s ease;
        border: 2px solid rgba(255,255,255,0.3);
      ">
        SELECT TEXT TO ANALYZE<br>
        <span style="font-size: 12px; opacity: 0.9;">Press ESC to exit • Min 10 characters</span>
      </div>
      <style>
        @keyframes slideInRight {
          from { transform: translateX(100%); opacity: 0; }
          to { transform: translateX(0); opacity: 1; }
        }
      </style>
    `;
    
    document.body.appendChild(selectionOverlay);
  }

  function removeSelectionOverlay() {
    if (selectionOverlay) {
      selectionOverlay.remove();
      selectionOverlay = null;
    }
  }

  function handleTextSelection(event) {
    if (!isSelectionMode) return;
    
    // Small delay to ensure selection is complete
    setTimeout(() => {
      const selectedText = window.getSelection().toString().trim();
      console.log('[CONTENT] Text selected:', selectedText);
      
      if (selectedText && selectedText.length >= 10) {
        showSelectionConfirmation(selectedText, event.clientX, event.clientY);
      } else if (selectedText && selectedText.length < 10) {
        // Show brief message for too short text
        showBriefMessage('Text too short (min 10 characters)', event.clientX, event.clientY);
      }
    }, 100);
  }

  function showBriefMessage(message, x, y) {
    const msgDiv = document.createElement('div');
    msgDiv.style.cssText = `
      position: fixed;
      left: ${Math.min(x, window.innerWidth - 200)}px;
      top: ${Math.min(y + 20, window.innerHeight - 60)}px;
      background: #ff9800;
      color: white;
      padding: 8px 12px;
      border-radius: 6px;
      font-family: Arial, sans-serif;
      font-size: 12px;
      z-index: 999999;
      box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    `;
    msgDiv.textContent = message;
    
    document.body.appendChild(msgDiv);
    
    setTimeout(() => {
      if (msgDiv.parentNode) msgDiv.remove();
    }, 2000);
  }

  function showSelectionConfirmation(text, x, y) {
    console.log('[CONTENT] Showing confirmation for text:', text.substring(0, 50) + '...');
    
    // Remove any existing confirmation
    const existingConfirm = document.getElementById('climate-detector-confirm');
    if (existingConfirm) existingConfirm.remove();
    
    const confirmDiv = document.createElement('div');
    confirmDiv.id = 'climate-detector-confirm';
    confirmDiv.innerHTML = `
      <div style="
        position: fixed;
        left: ${Math.min(x, window.innerWidth - 320)}px;
        top: ${Math.min(y + 20, window.innerHeight - 200)}px;
        background: white;
        border: 2px solid #4CAF50;
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.2);
        z-index: 999999;
        max-width: 300px;
        font-family: Arial, sans-serif;
      ">
        <div style="font-size: 14px; margin-bottom: 12px; color: #333;">
          <strong style="color: #4CAF50;">📖 Selected Text:</strong><br>
          <div style="
            background: #f5f5f5;
            padding: 10px;
            border-radius: 8px;
            margin: 8px 0;
            font-size: 12px;
            max-height: 80px;
            overflow-y: auto;
            color: #666;
            border-left: 3px solid #4CAF50;
          ">
            ${text.substring(0, 150)}${text.length > 150 ? '...' : ''}
          </div>
        </div>
        <div style="display: flex; gap: 10px;">
          <button id="climate-analyze-btn" style="
            flex: 1;
            background: #4CAF50;
            color: white;
            border: none;
            padding: 12px;
            border-radius: 8px;
            cursor: pointer;
            font-weight: bold;
            font-size: 13px;
            transition: background 0.3s;
          "> Analyze</button>
          <button id="climate-cancel-btn" style="
            flex: 1;
            background: #f44336;
            color: white;
            border: none;
            padding: 12px;
            border-radius: 8px;
            cursor: pointer;
            font-weight: bold;
            font-size: 13px;
            transition: background 0.3s;
          ">❌ Cancel</button>
        </div>
      </div>
    `;
    
    document.body.appendChild(confirmDiv);
    
    // Add hover effects
    const analyzeBtn = document.getElementById('climate-analyze-btn');
    const cancelBtn = document.getElementById('climate-cancel-btn');
    
    analyzeBtn.addEventListener('mouseenter', () => {
      analyzeBtn.style.background = '#45a049';
    });
    analyzeBtn.addEventListener('mouseleave', () => {
      analyzeBtn.style.background = '#4CAF50';
    });
    
    cancelBtn.addEventListener('mouseenter', () => {
      cancelBtn.style.background = '#da190b';
    });
    cancelBtn.addEventListener('mouseleave', () => {
      cancelBtn.style.background = '#f44336';
    });
    
    // Add button listeners
    analyzeBtn.addEventListener('click', () => {
      console.log('[CONTENT] Analyze button clicked');
      analyzeSelectedText(text);
      confirmDiv.remove();
    });
    
    cancelBtn.addEventListener('click', () => {
      console.log('[CONTENT] Cancel button clicked');
      confirmDiv.remove();
      window.getSelection().removeAllRanges();
      exitSelectionMode();
    });
    
    // Auto-remove after 15 seconds
    setTimeout(() => {
      if (confirmDiv.parentNode) {
        confirmDiv.remove();
      }
    }, 15000);
  }

  function analyzeSelectedText(text) {
    console.log('[CONTENT] Analyzing selected text:', text.substring(0, 50) + '...');
    
    // Send text to extension background script
    chrome.runtime.sendMessage({
      action: 'analyzeSelectedText',
      text: text,
      timestamp: Date.now()
    }).then((response) => {
      console.log('[CONTENT] Text sent to background script, response:', response);
    }).catch(error => {
      console.error('[CONTENT] Error sending text:', error);
    });
    
    // Clear selection and exit selection mode
    window.getSelection().removeAllRanges();
    exitSelectionMode();
    
    // Show success message
    const successDiv = document.createElement('div');
    successDiv.style.cssText = `
      position: fixed;
      top: 50%;
      left: 50%;
      transform: translate(-50%, -50%);
      background: #4CAF50;
      color: white;
      padding: 20px 30px;
      border-radius: 12px;
      font-family: Arial, sans-serif;
      font-size: 16px;
      font-weight: bold;
      z-index: 999999;
      box-shadow: 0 8px 24px rgba(0,0,0,0.3);
      text-align: center;
    `;
    successDiv.innerHTML = `
      ✅ Text Selected!<br>
      <span style="font-size: 12px; opacity: 0.9;">Opening analysis in side panel...</span>
    `;
    
    document.body.appendChild(successDiv);
    
    setTimeout(() => {
      if (successDiv.parentNode) {
        successDiv.remove();
      }
    }, 2000);
  }

  function handleEscapeKey(event) {
    if (event.key === 'Escape' && isSelectionMode) {
      console.log('[CONTENT] Escape key pressed, exiting selection mode');
      exitSelectionMode();
      window.getSelection().removeAllRanges();
      
      // Remove any confirmation dialogs
      const confirmDialog = document.getElementById('climate-detector-confirm');
      if (confirmDialog) confirmDialog.remove();
    }
  }

  // Clean up on page unload
  window.addEventListener('beforeunload', () => {
    exitSelectionMode();
  });

  function showOpenSidePanelPrompt(text) {
    console.log('[CONTENT] Showing side panel prompt for text:', text.substring(0, 50));
    
    // Remove any existing prompts
    const existingPrompt = document.getElementById('climate-sidepanel-prompt');
    if (existingPrompt) existingPrompt.remove();
    
    const promptDiv = document.createElement('div');
    promptDiv.id = 'climate-sidepanel-prompt';
    promptDiv.innerHTML = `
      <div style="
        position: fixed;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        background: white;
        border: 3px solid #4CAF50;
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 12px 32px rgba(0,0,0,0.3);
        z-index: 999999;
        max-width: 400px;
        font-family: Arial, sans-serif;
        text-align: center;
      ">
        <div style="font-size: 18px; font-weight: bold; color: #4CAF50; margin-bottom: 16px;">
          Text Selected Successfully!
        </div>
        <div style="font-size: 14px; color: #666; margin-bottom: 12px;">
          "${text}${text.length > 100 ? '...' : ''}"
        </div>
        <div style="font-size: 14px; color: #333; margin-bottom: 20px;">
          The side panel is not currently open.<br>
          Click below to open the analysis panel.
        </div>
        <div style="display: flex; gap: 12px;">
          <button id="climate-open-panel-btn" style="
            flex: 1;
            background: #4CAF50;
            color: white;
            border: none;
            padding: 12px 20px;
            border-radius: 8px;
            cursor: pointer;
            font-weight: bold;
            font-size: 14px;
          ">Open Analysis Panel</button>
          <button id="climate-dismiss-btn" style="
            flex: 1;
            background: #666;
            color: white;
            border: none;
            padding: 12px 20px;
            border-radius: 8px;
            cursor: pointer;
            font-weight: bold;
            font-size: 14px;
          ">Dismiss</button>
        </div>
      </div>
    `;
    
    document.body.appendChild(promptDiv);
    
    // Add button listeners
    document.getElementById('climate-open-panel-btn').addEventListener('click', () => {
      console.log('[CONTENT] User clicked open panel');
      promptDiv.remove();
      showExtensionIconInstruction();
    });
    
    document.getElementById('climate-dismiss-btn').addEventListener('click', () => {
      promptDiv.remove();
    });
    
    // Auto-remove after 15 seconds
    setTimeout(() => {
      if (promptDiv.parentNode) {
        promptDiv.remove();
      }
    }, 15000);
  }
  
  function showExtensionIconInstruction() {
    const instructionDiv = document.createElement('div');
    instructionDiv.style.cssText = `
      position: fixed;
      top: 20px;
      right: 20px;
      background: #2196F3;
      color: white;
      padding: 16px 20px;
      border-radius: 12px;
      font-family: Arial, sans-serif;
      font-size: 14px;
      font-weight: bold;
      z-index: 999999;
      box-shadow: 0 6px 20px rgba(0,0,0,0.3);
      max-width: 300px;
      animation: pulse 2s infinite;
    `;
    instructionDiv.innerHTML = `
      Click the Climate Detector extension icon<br>
      <span style="font-size: 12px; opacity: 0.9;">(Usually in the top-right corner of your browser)</span>
    `;
    
    // Add pulsing animation
    const style = document.createElement('style');
    style.textContent = `
      @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.05); }
      }
    `;
    document.head.appendChild(style);
    
    document.body.appendChild(instructionDiv);
    
    // Remove after 8 seconds
    setTimeout(() => {
      if (instructionDiv.parentNode) {
        instructionDiv.remove();
      }
    }, 8000);
  }

  // Initialize
  console.log('[CONTENT] Climate detector content script ready');
})();// Content script for text selection functionality
let isSelectionMode = false;
let selectionOverlay = null;

// Listen for messages from popup/sidepanel
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === 'toggleSelectionMode') {
    toggleSelectionMode();
    sendResponse({ success: true });
  } else if (request.action === 'getSelectedText') {
    const selectedText = window.getSelection().toString().trim();
    sendResponse({ selectedText: selectedText });
  } else if (request.action === 'analyzeSelectedText') {
    const selectedText = window.getSelection().toString().trim();
    if (selectedText) {
      // Send the selected text back to the extension
      chrome.runtime.sendMessage({
        action: 'analyzeText',
        text: selectedText
      });
      // Clear selection and exit selection mode
      window.getSelection().removeAllRanges();
      exitSelectionMode();
    }
    sendResponse({ selectedText: selectedText });
  }
});

function toggleSelectionMode() {
  if (isSelectionMode) {
    exitSelectionMode();
  } else {
    enterSelectionMode();
  }
}

function enterSelectionMode() {
  isSelectionMode = true;
  
  // Create overlay to indicate selection mode
  createSelectionOverlay();
  
  // Add event listeners
  document.addEventListener('mouseup', handleTextSelection);
  document.addEventListener('keydown', handleEscapeKey);
  
  // Change cursor to indicate selection mode
  document.body.style.cursor = 'crosshair';
  
  console.log('Selection mode activated');
}

function exitSelectionMode() {
  isSelectionMode = false;
  
  // Remove overlay
  removeSelectionOverlay();
  
  // Remove event listeners
  document.removeEventListener('mouseup', handleTextSelection);
  document.removeEventListener('keydown', handleEscapeKey);
  
  // Reset cursor
  document.body.style.cursor = '';
  
  console.log('Selection mode deactivated');
}

function createSelectionOverlay() {
  if (selectionOverlay) return;
  
  selectionOverlay = document.createElement('div');
  selectionOverlay.id = 'climate-detector-overlay';
  selectionOverlay.innerHTML = `
    <div style="
      position: fixed;
      top: 20px;
      right: 20px;
      background: #4CAF50;
      color: white;
      padding: 12px 20px;
      border-radius: 8px;
      font-family: Arial, sans-serif;
      font-size: 14px;
      font-weight: bold;
      z-index: 10000;
      box-shadow: 0 4px 12px rgba(0,0,0,0.3);
      animation: slideIn 0.3s ease;
    ">
      📖 Select text to analyze • Press ESC to exit
    </div>
    <style>
      @keyframes slideIn {
        from { transform: translateX(100%); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
      }
    </style>
  `;
  
  document.body.appendChild(selectionOverlay);
}

function removeSelectionOverlay() {
  if (selectionOverlay) {
    selectionOverlay.remove();
    selectionOverlay = null;
  }
}

function handleTextSelection(event) {
  if (!isSelectionMode) return;
  
  const selectedText = window.getSelection().toString().trim();
  
  if (selectedText && selectedText.length > 10) {
    // Show confirmation popup
    showSelectionConfirmation(selectedText, event.clientX, event.clientY);
  }
}

function showSelectionConfirmation(text, x, y) {
  // Remove any existing confirmation
  const existingConfirm = document.getElementById('climate-detector-confirm');
  if (existingConfirm) existingConfirm.remove();
  
  const confirmDiv = document.createElement('div');
  confirmDiv.id = 'climate-detector-confirm';
  confirmDiv.innerHTML = `
    <div style="
      position: fixed;
      left: ${Math.min(x, window.innerWidth - 300)}px;
      top: ${Math.min(y + 20, window.innerHeight - 150)}px;
      background: white;
      border: 1px solid #ddd;
      border-radius: 12px;
      padding: 16px;
      box-shadow: 0 8px 24px rgba(0,0,0,0.15);
      z-index: 10001;
      max-width: 280px;
      font-family: Arial, sans-serif;
    ">
      <div style="font-size: 14px; margin-bottom: 12px; color: #333;">
        <strong>Selected text:</strong><br>
        <div style="
          background: #f5f5f5;
          padding: 8px;
          border-radius: 6px;
          margin: 8px 0;
          font-size: 12px;
          max-height: 60px;
          overflow-y: auto;
          color: #666;
        ">
          ${text.substring(0, 100)}${text.length > 100 ? '...' : ''}
        </div>
      </div>
      <div style="display: flex; gap: 8px;">
        <button id="analyze-btn" style="
          flex: 1;
          background: #4CAF50;
          color: white;
          border: none;
          padding: 10px;
          border-radius: 6px;
          cursor: pointer;
          font-weight: bold;
          font-size: 12px;
        ">Analyze</button>
        <button id="cancel-btn" style="
          flex: 1;
          background: #f44336;
          color: white;
          border: none;
          padding: 10px;
          border-radius: 6px;
          cursor: pointer;
          font-weight: bold;
          font-size: 12px;
        ">Cancel</button>
      </div>
    </div>
  `;
  
  document.body.appendChild(confirmDiv);
  
  // Add button listeners
  document.getElementById('analyze-btn').addEventListener('click', () => {
    analyzeSelectedText(text);
    confirmDiv.remove();
  });
  
  document.getElementById('cancel-btn').addEventListener('click', () => {
    confirmDiv.remove();
    window.getSelection().removeAllRanges();
  });
  
  // Auto-remove after 10 seconds
  setTimeout(() => {
    if (confirmDiv.parentNode) {
      confirmDiv.remove();
    }
  }, 10000);
}

function analyzeSelectedText(text) {
  // Send text to extension for analysis
  chrome.runtime.sendMessage({
    action: 'analyzeText',
    text: text
  });
  
  // Clear selection and exit selection mode
  window.getSelection().removeAllRanges();
  exitSelectionMode();
  
  // Open side panel with the text
  chrome.runtime.sendMessage({
    action: 'openSidePanel',
    text: text
  });
}

function handleEscapeKey(event) {
  if (event.key === 'Escape' && isSelectionMode) {
    exitSelectionMode();
    window.getSelection().removeAllRanges();
  }
}

// Clean up on page unload
window.addEventListener('beforeunload', () => {
  exitSelectionMode();
});