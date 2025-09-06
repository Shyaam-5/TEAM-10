// Forms JavaScript - Form validation, submission, API mode

document.addEventListener('DOMContentLoaded', function() {
    initializeFormValidation();
    initializePasswordStrength();
    initializeReferralModes();
    initializeCharacterCounters();
    initializeFiltering();
});

// Form Validation
function initializeFormValidation() {
    const forms = document.querySelectorAll('form');
    
    forms.forEach(form => {
        // Real-time validation
        const inputs = form.querySelectorAll('input, textarea');
        inputs.forEach(input => {
            input.addEventListener('blur', () => validateField(input));
            input.addEventListener('input', () => clearFieldError(input));
        });

        // Form submission
        form.addEventListener('submit', function(e) {
            const isValid = validateForm(form);
            if (!isValid) {
                e.preventDefault();
                return false;
            }
            
            // Show loading state
            const submitBtn = form.querySelector('button[type="submit"]');
            if (submitBtn) {
                window.MedReferral.showLoading(submitBtn);
            }
        });
    });
}

function validateField(input) {
    const value = input.value.trim();
    const type = input.type;
    const required = input.hasAttribute('required');
    let isValid = true;
    let errorMessage = '';

    // Clear previous errors
    clearFieldError(input);

    // Required validation
    if (required && !value) {
        isValid = false;
        errorMessage = 'This field is required.';
    }
    
    // Type-specific validation
    if (value) {
        switch (type) {
            case 'email':
                const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
                if (!emailRegex.test(value)) {
                    isValid = false;
                    errorMessage = 'Please enter a valid email address.';
                }
                break;
                
            case 'password':
                if (input.name === 'password') {
                    if (value.length < 8) {
                        isValid = false;
                        errorMessage = 'Password must be at least 8 characters.';
                    }
                }
                break;
                
            case 'text':
                if (input.name === 'username') {
                    const usernameRegex = /^[A-Za-z0-9_]{3,30}$/;
                    if (!usernameRegex.test(value)) {
                        isValid = false;
                        errorMessage = 'Username must be 3-30 characters (letters, numbers, underscore only).';
                    }
                }
                break;
        }
    }

    // Show error if invalid
    if (!isValid) {
        showFieldError(input, errorMessage);
    }

    return isValid;
}

function validateForm(form) {
    const inputs = form.querySelectorAll('input[required], textarea[required]');
    let isFormValid = true;

    inputs.forEach(input => {
        const isFieldValid = validateField(input);
        if (!isFieldValid) {
            isFormValid = false;
        }
    });

    return isFormValid;
}

function showFieldError(input, message) {
    const errorElement = document.getElementById(input.id + '-error') || 
                        input.parentNode.querySelector('.form-error');
    
    if (errorElement) {
        errorElement.textContent = message;
        errorElement.style.display = 'block';
    }
    
    input.classList.add('error');
    input.setAttribute('aria-invalid', 'true');
}

function clearFieldError(input) {
    const errorElement = document.getElementById(input.id + '-error') || 
                        input.parentNode.querySelector('.form-error');
    
    if (errorElement) {
        errorElement.textContent = '';
        errorElement.style.display = 'none';
    }
    
    input.classList.remove('error');
    input.removeAttribute('aria-invalid');
}

// Password Strength Indicator
function initializePasswordStrength() {
    const passwordInput = document.getElementById('password');
    if (!passwordInput) return;

    passwordInput.addEventListener('input', function() {
        const password = this.value;
        const strength = calculatePasswordStrength(password);
        updatePasswordStrengthUI(strength);
    });
}

function calculatePasswordStrength(password) {
    let score = 0;
    
    if (password.length >= 8) score++;
    if (password.length >= 12) score++;
    if (/[a-z]/.test(password) && /[A-Z]/.test(password)) score++;
    if (/\d/.test(password)) score++;
    if (/[^A-Za-z0-9]/.test(password)) score++;
    
    return Math.min(score, 4);
}

function updatePasswordStrengthUI(strength) {
    const strengthFill = document.querySelector('.strength-fill');
    const strengthText = document.querySelector('.strength-text');
    
    if (!strengthFill || !strengthText) return;

    const strengthLabels = [
        'Enter password',
        'Very weak',
        'Weak', 
        'Good',
        'Strong'
    ];
    
    strengthFill.setAttribute('data-strength', strength);
    strengthText.textContent = strengthLabels[strength];
}

// Referral Mode Management
function initializeReferralModes() {
    const traditionalBtn = document.getElementById('traditional-mode');
    const apiBtn = document.getElementById('api-mode');
    const form = document.getElementById('referral-form');
    const apiResults = document.getElementById('api-results');
    
    if (!traditionalBtn || !apiBtn) return;

    traditionalBtn.addEventListener('click', () => switchMode('traditional'));
    apiBtn.addEventListener('click', () => switchMode('api'));

    function switchMode(mode) {
        if (mode === 'traditional') {
            traditionalBtn.classList.add('active');
            traditionalBtn.setAttribute('aria-pressed', 'true');
            apiBtn.classList.remove('active');
            apiBtn.setAttribute('aria-pressed', 'false');
            
            if (form) form.style.display = 'block';
            if (apiResults) apiResults.classList.add('hidden');
            
        } else if (mode === 'api') {
            apiBtn.classList.add('active');
            apiBtn.setAttribute('aria-pressed', 'true');
            traditionalBtn.classList.remove('active');
            traditionalBtn.setAttribute('aria-pressed', 'false');
            
            if (form) form.style.display = 'none';
            if (apiResults) apiResults.classList.remove('hidden');
            
            setupApiMode();
        }
    }
}

function setupApiMode() {
    const textarea = document.getElementById('patient_note');
    const resultsContainer = document.querySelector('#api-results .results-content');
    const loadingContainer = document.querySelector('#api-results .results-loading');
    
    if (!textarea || !resultsContainer || !loadingContainer) return;

    const debouncedAnalyze = window.MedReferral.debounce(analyzeNote, 1500);
    
    textarea.addEventListener('input', function() {
        const note = this.value.trim();
        if (note.length > 50) { // Minimum length before triggering
            debouncedAnalyze(note);
        } else {
            resultsContainer.classList.add('hidden');
            loadingContainer.classList.remove('hidden');
        }
    });
}

async function analyzeNote(note) {
    const resultsContainer = document.querySelector('#api-results .results-content');
    const loadingContainer = document.querySelector('#api-results .results-loading');
    
    // Show loading
    resultsContainer.classList.add('hidden');
    loadingContainer.classList.remove('hidden');

    try {
        const response = await fetch('/api/referral', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ patient_note: note })
        });

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        const data = await response.json();
        renderApiResults(data, resultsContainer);
        
        // Show results
        loadingContainer.classList.add('hidden');
        resultsContainer.classList.remove('hidden');
        
    } catch (error) {
        console.error('API call failed:', error);
        renderApiError(error.message, resultsContainer);
        loadingContainer.classList.add('hidden');
        resultsContainer.classList.remove('hidden');
    }
}

function renderApiResults(data, container) {
    const { recommended, distribution, location } = data;
    
    const html = `
        <div class="api-result-card card">
            <div class="card-header">
                <h3 class="text-lg font-semibold">Live Analysis Results</h3>
            </div>
            <div class="card-body">
                <div class="api-recommendation">
                    <div class="recommendation-badge">
                        <span class="badge badge-primary">${recommended}</span>
                        <span class="recommendation-confidence">${distribution[recommended]}% confidence</span>
                    </div>
                    
                    <div class="location-info mt-4">
                        <strong>Detected Location:</strong> 
                        <span class="${location === 'Unknown' ? 'text-neutral-500' : 'text-secondary-600'}">${location}</span>
                    </div>
                </div>
                
                <div class="api-actions mt-6">
                    <form method="POST" action="/referral" style="display: inline;">
                        <input type="hidden" name="patient_note" value="${document.getElementById('patient_note').value}">
                        <button type="submit" class="btn btn-primary">Get Full Analysis</button>
                    </form>
                </div>
            </div>
        </div>
    `;
    
    container.innerHTML = html;
}

function renderApiError(errorMessage, container) {
    const html = `
        <div class="api-error-card card">
            <div class="card-body">
                <div class="empty-state">
                    <svg class="empty-state-icon" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <circle cx="12" cy="12" r="10" stroke="currentColor" stroke-width="2"/>
                        <path d="M12 8V12M12 16H12.01" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
                    </svg>
                    <h4 class="empty-state-title">Analysis Error</h4>
                    <p class="empty-state-text">${errorMessage}</p>
                </div>
            </div>
        </div>
    `;
    
    container.innerHTML = html;
}

// Character Counters
function initializeCharacterCounters() {
    const textareas = document.querySelectorAll('textarea[maxlength]');
    
    textareas.forEach(textarea => {
        // Initialize counter
        window.MedReferral.updateCharCounter(textarea);
        
        // Update on input
        textarea.addEventListener('input', function() {
            window.MedReferral.updateCharCounter(this);
        });
    });
}

// Client-side Filtering (for ranking page)
function initializeFiltering() {
    const searchInput = document.getElementById('search-filter');
    const distanceSlider = document.getElementById('distance-filter');
    const distanceValue = document.getElementById('distance-value');
    
    if (searchInput) {
        const debouncedFilter = window.MedReferral.debounce(filterClinics, 300);
        searchInput.addEventListener('input', debouncedFilter);
    }
    
    if (distanceSlider && distanceValue) {
        distanceSlider.addEventListener('input', function() {
            distanceValue.textContent = `${this.value} km`;
            filterClinics();
        });
    }
}

function filterClinics() {
    const searchTerm = document.getElementById('search-filter')?.value.toLowerCase() || '';
    const maxDistance = parseFloat(document.getElementById('distance-filter')?.value) || 100;
    const clinicCards = document.querySelectorAll('.clinic-card');
    const noResults = document.getElementById('no-results');
    
    let visibleCount = 0;
    
    clinicCards.forEach(card => {
        const name = card.dataset.name || '';
        const address = card.dataset.address || '';
        const distance = parseFloat(card.dataset.distance) || 0;
        
        const matchesSearch = !searchTerm || 
                            name.includes(searchTerm) || 
                            address.includes(searchTerm);
        const withinDistance = distance <= maxDistance;
        
        if (matchesSearch && withinDistance) {
            card.style.display = 'block';
            visibleCount++;
        } else {
            card.style.display = 'none';
        }
    });
    
    // Show/hide no results message
    if (noResults) {
        if (visibleCount === 0 && clinicCards.length > 0) {
            noResults.classList.remove('hidden');
        } else {
            noResults.classList.add('hidden');
        }
    }
}

// Enhanced form submissions with better error handling
function handleFormSubmission(form, endpoint, options = {}) {
    return new Promise((resolve, reject) => {
        const formData = new FormData(form);
        const submitBtn = form.querySelector('button[type="submit"]');
        
        if (submitBtn) {
            window.MedReferral.showLoading(submitBtn);
        }
        
        fetch(endpoint, {
            method: 'POST',
            body: formData,
            ...options
        })
        .then(response => {
            if (submitBtn) {
                window.MedReferral.hideLoading(submitBtn);
            }
            
            if (response.ok) {
                resolve(response);
            } else {
                reject(new Error(`HTTP ${response.status}: ${response.statusText}`));
            }
        })
        .catch(error => {
            if (submitBtn) {
                window.MedReferral.hideLoading(submitBtn);
            }
            reject(error);
        });
    });
}

// Input animations and enhancements
function addInputEnhancements() {
    const inputs = document.querySelectorAll('.form-input');
    
    inputs.forEach(input => {
        // Floating label effect (if using floating labels)
        input.addEventListener('focus', function() {
            this.parentNode.classList.add('focused');
        });
        
        input.addEventListener('blur', function() {
            if (!this.value) {
                this.parentNode.classList.remove('focused');
            }
        });
        
        // Error shake animation
        input.addEventListener('invalid', function() {
            this.classList.add('shake');
            setTimeout(() => {
                this.classList.remove('shake');
            }, 500);
        });
    });
}

// Initialize input enhancements
document.addEventListener('DOMContentLoaded', addInputEnhancements);

// Export for use in other scripts
window.FormUtils = {
    validateField,
    validateForm,
    handleFormSubmission,
    showFieldError,
    clearFieldError
};