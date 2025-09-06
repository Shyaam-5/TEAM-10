// Dr. Ellie Chatbot JavaScript

class ChatbotWidget {
    constructor() {
        this.isOpen = false;
        this.isTyping = false;
        this.messageHistory = [];
        this.initializeElements();
        this.bindEvents();
        this.loadWelcomeMessage();
    }

    initializeElements() {
        this.widget = document.getElementById('chatbot-widget');
        this.button = document.getElementById('chatbot-button');
        this.window = document.getElementById('chatbot-window');
        this.closeBtn = document.getElementById('chatbot-close');
        this.messagesContainer = document.getElementById('chatbot-messages');
        this.input = document.getElementById('chatbot-input');
        this.sendBtn = document.getElementById('chatbot-send');
        this.typingIndicator = document.getElementById('chatbot-typing');
        this.notification = document.getElementById('chatbot-notification');
    }

    bindEvents() {
        // Toggle chat window
        this.button.addEventListener('click', () => this.toggleChat());
        this.closeBtn.addEventListener('click', () => this.closeChat());

        // Send message
        this.sendBtn.addEventListener('click', () => this.sendMessage());
        this.input.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });

        // Input validation
        this.input.addEventListener('input', () => this.validateInput());

        // Close on outside click
        document.addEventListener('click', (e) => {
            if (this.isOpen && !this.widget.contains(e.target)) {
                this.closeChat();
            }
        });

        // Auto-resize input
        this.input.addEventListener('input', () => this.autoResizeInput());
    }

    toggleChat() {
        if (this.isOpen) {
            this.closeChat();
        } else {
            this.openChat();
        }
    }

    openChat() {
        this.isOpen = true;
        this.window.style.display = 'flex';
        this.button.style.display = 'none';
        this.hideNotification();
        this.input.focus();
        this.scrollToBottom();
    }

    closeChat() {
        this.isOpen = false;
        this.window.style.display = 'none';
        this.button.style.display = 'flex';
    }

    validateInput() {
        const message = this.input.value.trim();
        this.sendBtn.disabled = !message || this.isTyping;
    }

    autoResizeInput() {
        this.input.style.height = 'auto';
        this.input.style.height = Math.min(this.input.scrollHeight, 100) + 'px';
    }

    async sendMessage() {
        const message = this.input.value.trim();
        if (!message || this.isTyping) return;

        // Add user message
        this.addMessage(message, 'user');
        this.input.value = '';
        this.validateInput();
        this.autoResizeInput();

        // Show typing indicator
        this.showTyping();

        try {
            const response = await fetch('/api/chatbot/ask', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ question: message })
            });

            const data = await response.json();
            
            if (response.ok) {
                this.addMessage(data.answer, 'bot');
            } else {
                this.addMessage('I apologize, but I encountered an error. Please try again.', 'bot');
            }
        } catch (error) {
            console.error('Chatbot error:', error);
            this.addMessage('I apologize, but I\'m having trouble connecting right now. Please try again later.', 'bot');
        } finally {
            this.hideTyping();
        }
    }

    addMessage(content, sender) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `chatbot-message chatbot-message-${sender}`;

        const avatarDiv = document.createElement('div');
        avatarDiv.className = 'chatbot-message-avatar';
        
        if (sender === 'bot') {
            avatarDiv.innerHTML = `
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path d="M20 21V19C20 17.9391 19.5786 16.9217 18.8284 16.1716C18.0783 15.4214 17.0609 15 16 15H8C6.93913 15 5.92172 15.4214 5.17157 16.1716C4.42143 16.9217 4 17.9391 4 19V21" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                    <circle cx="12" cy="7" r="4" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                </svg>
            `;
        } else {
            avatarDiv.innerHTML = `
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path d="M20 21V19C20 17.9391 19.5786 16.9217 18.8284 16.1716C18.0783 15.4214 17.0609 15 16 15H8C6.93913 15 5.92172 15.4214 5.17157 16.1716C4.42143 16.9217 4 17.9391 4 19V21" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                    <circle cx="12" cy="7" r="4" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                </svg>
            `;
        }

        const contentDiv = document.createElement('div');
        contentDiv.className = 'chatbot-message-content';
        contentDiv.innerHTML = `<p>${this.escapeHtml(content)}</p>`;

        messageDiv.appendChild(avatarDiv);
        messageDiv.appendChild(contentDiv);

        this.messagesContainer.appendChild(messageDiv);
        this.scrollToBottom();

        // Store in history
        this.messageHistory.push({ content, sender, timestamp: new Date() });
    }

    showTyping() {
        this.isTyping = true;
        this.typingIndicator.style.display = 'flex';
        this.validateInput();
        this.scrollToBottom();
    }

    hideTyping() {
        this.isTyping = false;
        this.typingIndicator.style.display = 'none';
        this.validateInput();
    }

    scrollToBottom() {
        setTimeout(() => {
            this.messagesContainer.scrollTop = this.messagesContainer.scrollHeight;
        }, 100);
    }

    showNotification() {
        this.notification.style.display = 'flex';
    }

    hideNotification() {
        this.notification.style.display = 'none';
    }

    async loadWelcomeMessage() {
        try {
            const response = await fetch('/api/chatbot/welcome');
            const data = await response.json();
            
            if (response.ok) {
                // Update the initial welcome message
                const welcomeMessage = this.messagesContainer.querySelector('.chatbot-message-bot .chatbot-message-content p');
                if (welcomeMessage) {
                    welcomeMessage.textContent = data.answer;
                }
            }
        } catch (error) {
            console.error('Error loading welcome message:', error);
        }
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    // Public methods for external control
    open() {
        this.openChat();
    }

    close() {
        this.closeChat();
    }

    sendMessageExternal(text) {
        this.input.value = text;
        this.sendMessage();
    }

    showNotificationBadge() {
        this.showNotification();
    }
}

// Initialize chatbot when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    window.chatbot = new ChatbotWidget();
    
    // Optional: Show notification after a delay to encourage interaction
    setTimeout(() => {
        if (!window.chatbot.isOpen) {
            window.chatbot.showNotification();
        }
    }, 5000);
});

// Export for potential external use
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ChatbotWidget;
}
