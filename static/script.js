// Global variables
let isConnected = false;
let modelInfo = null;
let chatHistory = [];
let isGenerating = false;

// DOM elements
const statusDot = document.getElementById('statusDot');
const statusText = document.getElementById('statusText');
const chatMessages = document.getElementById('chatMessages');
const messageInput = document.getElementById('messageInput');
const sendButton = document.getElementById('sendButton');
const charCount = document.getElementById('charCount');
const modelInfoElement = document.getElementById('modelInfo');
const settingsPanel = document.getElementById('settingsPanel');
const loadingOverlay = document.getElementById('loadingOverlay');
const toastContainer = document.getElementById('toastContainer');
const maxLengthSlider = document.getElementById('maxLength');
const maxLengthValue = document.getElementById('maxLengthValue');
const showTypingCheckbox = document.getElementById('showTyping');

// API Configuration
const API_BASE = window.location.origin;
const themeToggle = document.getElementById('themeToggle');

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
    setupEventListeners();
    checkAPIHealth();
    initTheme();
});

function initializeApp() {
    // Auto-resize textarea
    messageInput.addEventListener('input', function() {
        this.style.height = 'auto';
        this.style.height = Math.min(this.scrollHeight, 120) + 'px';
        
        // Update character count
        const count = this.value.length;
        charCount.textContent = count;
        
        // Enable/disable send button
        sendButton.disabled = count === 0 || isGenerating;
        
        // Color character counter based on limit
        if (count > 450) {
            charCount.style.color = 'var(--error-color)';
        } else if (count > 400) {
            charCount.style.color = 'var(--warning-color)';
        } else {
            charCount.style.color = 'var(--text-muted)';
        }
    });
    
    // Settings slider
    maxLengthSlider.addEventListener('input', function() {
        maxLengthValue.textContent = this.value;
    });
    
    // Enter key to send message
    messageInput.addEventListener('keydown', function(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            if (!sendButton.disabled) {
                sendMessage();
            }
        }
    });
}

function setupEventListeners() {
    sendButton.addEventListener('click', sendMessage);
    
    // Theme toggle
    if (themeToggle) {
        themeToggle.addEventListener('click', toggleTheme);
    }
    
    // Close settings panel when clicking outside
    document.addEventListener('click', function(e) {
        if (!settingsPanel.contains(e.target) && !e.target.closest('.settings-btn')) {
            settingsPanel.classList.remove('show');
        }
    });
}

function initTheme() {
    const pref = localStorage.getItem('chat_theme') || 'dark';
    setTheme(pref);
}

function toggleTheme() {
    const isLight = document.body.classList.contains('theme-light');
    setTheme(isLight ? 'dark' : 'light');
}

function setTheme(theme) {
    if (theme === 'light') {
        document.body.classList.add('theme-light');
    } else {
        document.body.classList.remove('theme-light');
    }
    localStorage.setItem('chat_theme', theme);
}

async function checkAPIHealth() {
    try {
        showToast('正在连接API服务器...', 'info');
        
        const response = await fetch(`${API_BASE}/health`);
        const data = await response.json();
        
        if (response.ok && data.model_loaded) {
            setConnectionStatus(true, '已连接');
            modelInfo = data;
            updateModelInfo(data);
            showToast('✅ API连接成功！', 'success');
            
            // Load model info
            await loadModelInfo();
        } else {
            setConnectionStatus(false, '模型未加载');
            showToast('⚠️ 模型未加载，请检查服务器状态', 'warning');
        }
    } catch (error) {
        console.error('Health check failed:', error);
        setConnectionStatus(false, '连接失败');
        showToast('❌ 无法连接到API服务器', 'error');
    }
}

async function loadModelInfo() {
    try {
        const response = await fetch(`${API_BASE}/model-info`);
        const data = await response.json();
        
        if (response.ok) {
            const info = `${data.model_type.toUpperCase()} | ${data.vocab_size} 词汇 | ${data.device.toUpperCase()}`;
            modelInfoElement.innerHTML = `<i class="fas fa-microchip"></i> ${info}`;
        }
    } catch (error) {
        console.error('Failed to load model info:', error);
    }
}

function setConnectionStatus(connected, text) {
    isConnected = connected;
    statusText.textContent = text;
    statusDot.className = 'status-dot ' + (connected ? 'connected' : 'error');
}

function updateModelInfo(data) {
    const info = `${data.model_type.toUpperCase()} 模型 | 词汇量: ${data.vocab_size} | 设备: ${data.device.toUpperCase()}`;
    modelInfoElement.innerHTML = `<i class="fas fa-microchip"></i> ${info}`;
}

async function sendMessage() {
    const message = messageInput.value.trim();
    if (!message || isGenerating || !isConnected) return;
    
    // Add user message to chat
    addMessage('user', message);
    
    // Clear input
    messageInput.value = '';
    messageInput.style.height = 'auto';
    charCount.textContent = '0';
    sendButton.disabled = true;
    
    // Set generating state
    isGenerating = true;
    
    // Show typing indicator
    const typingId = showTypingIndicator();
    
    try {
        // Call API
        const response = await fetch(`${API_BASE}/generate`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                prompt: message,
                max_length: parseInt(maxLengthSlider.value)
            })
        });
        
        const data = await response.json();
        
        // Remove typing indicator
        removeTypingIndicator(typingId);
        
        if (response.ok) {
            // Add assistant response
            const assistantMessage = data.generated_text || '抱歉，我无法生成回复。';
            
            if (showTypingCheckbox.checked) {
                await addMessageWithTyping('assistant', assistantMessage);
            } else {
                addMessage('assistant', assistantMessage);
            }
            
            // Update chat history
            chatHistory.push({
                user: message,
                assistant: assistantMessage,
                timestamp: new Date().toISOString()
            });
            
        } else {
            addMessage('assistant', `❌ 生成失败: ${data.detail || '未知错误'}`);
            showToast('生成失败，请重试', 'error');
        }
        
    } catch (error) {
        console.error('Generation failed:', error);
        removeTypingIndicator(typingId);
        addMessage('assistant', '❌ 网络错误，请检查连接后重试。');
        showToast('网络错误，请重试', 'error');
    } finally {
        isGenerating = false;
        sendButton.disabled = false;
        messageInput.focus();
    }
}

function addMessage(sender, text) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}`;
    
    const avatarDiv = document.createElement('div');
    avatarDiv.className = 'message-avatar';
    avatarDiv.innerHTML = sender === 'user' ? '<i class="fas fa-user"></i>' : '<i class="fas fa-robot"></i>';
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    
    const textDiv = document.createElement('div');
    textDiv.className = 'message-text';
    textDiv.textContent = text;
    
    const timeDiv = document.createElement('div');
    timeDiv.className = 'message-time';
    timeDiv.textContent = new Date().toLocaleTimeString('zh-CN');
    
    contentDiv.appendChild(textDiv);
    contentDiv.appendChild(timeDiv);
    messageDiv.appendChild(avatarDiv);
    messageDiv.appendChild(contentDiv);
    
    // Remove welcome message if it exists
    const welcomeMessage = chatMessages.querySelector('.welcome-message');
    if (welcomeMessage) {
        welcomeMessage.remove();
    }
    
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    
    return messageDiv;
}

async function addMessageWithTyping(sender, text) {
    const messageDiv = addMessage(sender, '');
    const textDiv = messageDiv.querySelector('.message-text');
    
    // Typing effect
    let currentText = '';
    const typingSpeed = 30; // milliseconds per character
    
    for (let i = 0; i < text.length; i++) {
        currentText += text[i];
        textDiv.textContent = currentText;
        chatMessages.scrollTop = chatMessages.scrollHeight;
        await new Promise(resolve => setTimeout(resolve, typingSpeed));
    }
}

function showTypingIndicator() {
    const typingId = 'typing-' + Date.now();
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message assistant';
    messageDiv.id = typingId;
    
    const avatarDiv = document.createElement('div');
    avatarDiv.className = 'message-avatar';
    avatarDiv.innerHTML = '<i class="fas fa-robot"></i>';
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    
    const typingDiv = document.createElement('div');
    typingDiv.className = 'typing-indicator';
    typingDiv.innerHTML = '<div class="typing-dot"></div><div class="typing-dot"></div><div class="typing-dot"></div>';
    
    contentDiv.appendChild(typingDiv);
    messageDiv.appendChild(avatarDiv);
    messageDiv.appendChild(contentDiv);
    
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    
    return typingId;
}

function removeTypingIndicator(typingId) {
    const typingElement = document.getElementById(typingId);
    if (typingElement) {
        typingElement.remove();
    }
}

function askSample(question) {
    messageInput.value = question;
    messageInput.focus();
    messageInput.dispatchEvent(new Event('input'));
    
    // Auto-send if connected
    if (isConnected && !isGenerating) {
        setTimeout(() => sendMessage(), 500);
    }
}

function toggleSettings() {
    settingsPanel.classList.toggle('show');
}

function clearChat() {
    chatMessages.innerHTML = `
        <div class="welcome-message">
            <div class="welcome-icon">
                <i class="fas fa-gavel"></i>
            </div>
            <h2>欢迎使用中国法律RAG智能助手</h2>
            <p>我是基于RNN模型训练的中国法律文本生成AI助手。您可以向我咨询法律问题，我会基于训练数据为您提供相关的法律分析和建议。</p>
            <div class="sample-questions">
                <h3>示例问题：</h3>
                <div class="sample-buttons">
                    <button class="sample-btn" onclick="askSample('王军的行为是否符合中国刑法关于盗窃罪的构成要件')">
                        <i class="fas fa-question-circle"></i>
                        盗窃罪构成要件分析
                    </button>
                    <button class="sample-btn" onclick="askSample('根据《刑法》第二百六十四条，本案适用的量刑幅度是什么')">
                        <i class="fas fa-book"></i>
                        量刑幅度咨询
                    </button>
                    <button class="sample-btn" onclick="askSample('王军主动认罪并退还赃物，是否应当对量刑产生影响？')">
                        <i class="fas fa-handshake"></i>
                        认罪态度影响
                    </button>
                    <button class="sample-btn" onclick="askSample('你对此案的法律意见或推荐的处理结果是什么')">
                        <i class="fas fa-lightbulb"></i>
                        法律意见建议
                    </button>
                </div>
            </div>
        </div>
    `;
    
    chatHistory = [];
    settingsPanel.classList.remove('show');
    showToast('对话已清空', 'success');
}

function showToast(message, type = 'info') {
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    
    const icon = type === 'success' ? 'fas fa-check-circle' : 
                 type === 'error' ? 'fas fa-exclamation-circle' :
                 type === 'warning' ? 'fas fa-exclamation-triangle' :
                 'fas fa-info-circle';
    
    toast.innerHTML = `
        <i class="${icon}"></i>
        <span>${message}</span>
    `;
    
    toastContainer.appendChild(toast);
    
    // Auto-remove after 5 seconds
    setTimeout(() => {
        if (toast.parentNode) {
            toast.parentNode.removeChild(toast);
        }
    }, 5000);
}

function showLoading(show = true) {
    if (show) {
        loadingOverlay.classList.add('show');
    } else {
        loadingOverlay.classList.remove('show');
    }
}

// Utility function to format text (could be expanded for markdown support)
function formatText(text) {
    // Basic text formatting - could be enhanced
    return text.replace(/\n/g, '<br>');
}

// Auto-reconnect functionality
setInterval(async () => {
    if (!isConnected) {
        await checkAPIHealth();
    }
}, 30000); // Check every 30 seconds

// Handle page visibility for better UX
document.addEventListener('visibilitychange', () => {
    if (!document.hidden && !isConnected) {
        checkAPIHealth();
    }
});

// Handle window focus
window.addEventListener('focus', () => {
    if (!isConnected) {
        checkAPIHealth();
    }
});
