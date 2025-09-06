// Session management utilities
class SessionManager {
    constructor() {
        this.checkInterval = null;
        this.init();
    }

    init() {
        this.checkSessionStatus();
        this.startPeriodicCheck();
        this.setupEventListeners();
    }

    async checkSessionStatus() {
        try {
            const response = await fetch('/api/session-status');
            const data = await response.json();
            
            if (data.logged_in) {
                this.updateUIForLoggedIn(data.user);
            } else {
                this.updateUIForLoggedOut();
            }
        } catch (error) {
            console.error('Session check failed:', error);
            this.updateUIForLoggedOut();
        }
    }

    updateUIForLoggedIn(user) {
        const authActions = document.querySelector('.auth-actions');
        if (authActions) {
            authActions.innerHTML = `
                <span class="user-welcome">Welcome, ${user.username}!</span>
                <a href="/referral" class="btn btn-primary">Dashboard</a>
                <a href="/logout" class="btn btn-outline">Log Out</a>
            `;
        }

        // Update hero actions if on landing page
        const heroActions = document.querySelector('.hero-actions');
        if (heroActions) {
            const tryButton = heroActions.querySelector('a[href="/referral"]');
            if (tryButton) {
                tryButton.textContent = 'Go to Dashboard';
            }
        }
    }

    updateUIForLoggedOut() {
        const authActions = document.querySelector('.auth-actions');
        if (authActions) {
            authActions.innerHTML = `
                <a href="/login" class="btn btn-outline">Log In</a>
                <a href="/signup" class="btn btn-primary">Sign Up</a>
            `;
        }
    }

    startPeriodicCheck() {
        // Check session every 5 minutes
        this.checkInterval = setInterval(() => {
            this.checkSessionStatus();
        }, 5 * 60 * 1000);
    }

    setupEventListeners() {
        // Check session when page becomes visible
        document.addEventListener('visibilitychange', () => {
            if (!document.hidden) {
                this.checkSessionStatus();
            }
        });

        // Check session when window gains focus
        window.addEventListener('focus', () => {
            this.checkSessionStatus();
        });
    }

    destroy() {
        if (this.checkInterval) {
            clearInterval(this.checkInterval);
        }
    }
}

// Initialize session manager when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.sessionManager = new SessionManager();
});
