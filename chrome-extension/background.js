// Background service worker

// Handle messages from content script, popup, and sidepanel
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  console.log('[BACKGROUND] Message received:', request);
  
  if (request.action === 'analyzeSelectedText') {
    handleTextSelection(request, sender);
    sendResponse({ success: true });
  } else if (request.action === 'prepareForTextSelection') {
    // Side panel is preparing for text selection
    console.log('[BACKGROUND] Side panel prepared for text selection');
    sendResponse({ success: true });
  } else if (request.action === 'activateTextSelection') {
    // Handle text selection activation from popup
    handleTextSelectionActivation(request.tabId);
    sendResponse({ success: true });
  } else if (request.action === 'openSidePanel') {
    // Open side panel
    if (sender.tab) {
      chrome.sidePanel.open({ tabId: sender.tab.id });
    }
    sendResponse({ success: true });
  }
  
  return true; // Keep message channel open for async response
});

async function handleTextSelectionActivation(tabId) {
  console.log('[BACKGROUND] Activating text selection for tab:', tabId);
  
  try {
    // Try to send message to existing content script first
    await chrome.tabs.sendMessage(tabId, { action: 'enterSelectionMode' });
    console.log('[BACKGROUND] Message sent to existing content script');
  } catch (error) {
    console.log('[BACKGROUND] No existing content script, injecting new one');
    
    // If no content script exists, inject one
    try {
      await chrome.scripting.executeScript({
        target: { tabId: tabId },
        func: enterSelectionModeDirectly
      });
      console.log('[BACKGROUND] Direct selection mode injection successful');
    } catch (injectionError) {
      console.error('[BACKGROUND] Failed to inject selection mode:', injectionError);
    }
  }
}

// Function to inject directly into the page
function enterSelectionModeDirectly() {
  console.log('[INJECTED] Entering selection mode directly');
  
  let isSelectionMode = false;
  let selectionOverlay = null;
  
  function createSelectionOverlay() {
    if (selectionOverlay) return;
    
    selectionOverlay = document.createElement('div');
    selectionOverlay.id = 'climate-detector-overlay-direct';
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
        border: 2px solid rgba(255,255,255,0.3);
        cursor: pointer;
      ">
        SELECTION MODE ACTIVE<br>
        <span style="font-size: 12px; opacity: 0.9;">Select text to analyze • Click here to exit</span>
      </div>
    `;
    
    document.body.appendChild(selectionOverlay);
    
    // Click to exit
    selectionOverlay.addEventListener('click', () => {
      exitSelectionMode();
    });
  }
  
  function exitSelectionMode() {
    if (selectionOverlay) {
      selectionOverlay.remove();
      selectionOverlay = null;
    }
    document.body.style.cursor = '';
    document.removeEventListener('mouseup', handleSelection);
    document.removeEventListener('keydown', handleEscape);
    isSelectionMode = false;
    console.log('[INJECTED] Selection mode exited');
  }
  
  function handleSelection(event) {
    setTimeout(() => {
      const selectedText = window.getSelection().toString().trim();
      if (selectedText && selectedText.length >= 10) {
        showConfirmation(selectedText, event.clientX, event.clientY);
      }
    }, 100);
  }
  
  function handleEscape(event) {
    if (event.key === 'Escape') {
      exitSelectionMode();
    }
  }
  
  function showConfirmation(text, x, y) {
    const confirmDiv = document.createElement('div');
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
          <strong style="color: #4CAF50;">Selected Text:</strong><br>
          <div style="background: #f5f5f5; padding: 10px; border-radius: 8px; margin: 8px 0; font-size: 12px; max-height: 80px; overflow-y: auto; color: #666; border-left: 3px solid #4CAF50;">
            ${text.substring(0, 150)}${text.length > 150 ? '...' : ''}
          </div>
        </div>
        <div style="display: flex; gap: 10px;">
          <button onclick="analyzeText('${text.replace(/'/g, "\\'")}'); this.parentElement.parentElement.parentElement.remove();" style="flex: 1; background: #4CAF50; color: white; border: none; padding: 12px; border-radius: 8px; cursor: pointer; font-weight: bold; font-size: 13px;">✅ Analyze</button>
          <button onclick="this.parentElement.parentElement.parentElement.remove(); window.getSelection().removeAllRanges();" style="flex: 1; background: #f44336; color: white; border: none; padding: 12px; border-radius: 8px; cursor: pointer; font-weight: bold; font-size: 13px;">❌ Cancel</button>
        </div>
      </div>
    `;
    document.body.appendChild(confirmDiv);
  }
  
  // Global function for analyze button
  window.analyzeText = function(text) {
    console.log('[INJECTED] Sending text for analysis:', text.substring(0, 50));
    if (window.chrome && window.chrome.runtime) {
      window.chrome.runtime.sendMessage({
        action: 'analyzeSelectedText',
        text: text,
        timestamp: Date.now()
      });
    }
    exitSelectionMode();
  };
  
  // Start selection mode
  isSelectionMode = true;
  document.body.style.cursor = 'crosshair';
  createSelectionOverlay();
  document.addEventListener('mouseup', handleSelection);
  document.addEventListener('keydown', handleEscape);
  
  console.log('[INJECTED] Selection mode activated');
}

function forwardToContentScript(request, sender) {
  if (sender.tab) {
    chrome.tabs.sendMessage(sender.tab.id, request).catch(() => {
      console.log('[BACKGROUND] Could not forward to content script');
    });
  }
}

async function handleTextSelection(request, sender) {
  console.log('[BACKGROUND] Handling text selection:', request.text.substring(0, 50) + '...');
  
  try {
    // Store the selected text with current timestamp
    await chrome.storage.local.set({ 
      pendingAnalysis: request.text,
      timestamp: Date.now(), // Always use current timestamp
      autoAnalyze: true
    });
    
    console.log('[BACKGROUND] Text stored with timestamp:', Date.now());
    
    // Try to send message to any open side panels
    let messageSent = false;
    try {
      await chrome.runtime.sendMessage({
        action: 'textSelectedForAnalysis',
        text: request.text,
        tabId: sender.tab?.id
      });
      console.log('[BACKGROUND] Message sent successfully to open side panel');
      messageSent = true;
    } catch (messageError) {
      console.log('[BACKGROUND] No open side panel found:', messageError.message);
    }
    
    if (!messageSent) {
      // If no side panel is open, show prompt to open it
      if (sender.tab) {
        chrome.tabs.sendMessage(sender.tab.id, {
          action: 'showOpenSidePanelPrompt',
          text: request.text.substring(0, 100)
        }).catch(() => {
          console.log('[BACKGROUND] Could not send prompt to content script');
        });
      }
    }
    
  } catch (error) {
    console.error('[BACKGROUND] Error handling text selection:', error);
  }
}


// Handle extension icon click to open side panel
chrome.action.onClicked.addListener((tab) => {
  console.log('[BACKGROUND] Extension icon clicked, opening side panel');
  chrome.sidePanel.open({ tabId: tab.id });
});

// Handle installation
chrome.runtime.onInstalled.addListener(() => {
  console.log('[BACKGROUND] Climate Misinformation Detector installed');
});