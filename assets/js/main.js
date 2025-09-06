// Main JavaScript - Core functionality, modals, flash messages, navigation

document.addEventListener('DOMContentLoaded', function() {
    // Initialize all main components
    initializeFlashMessages();
    initializeModals();
    initializeNavigation();
    initializeSampleNotes();
    initializeCopyButtons();
    initializeProgressAnimations();
});

// Flash Messages Management
function initializeFlashMessages() {
    const flashContainer = document.getElementById('flash-messages');
    if (!flashContainer) return;

    // Auto-dismiss flash messages after 5 seconds
    const flashMessages = flashContainer.querySelectorAll('.flash');
    flashMessages.forEach(flash => {
        // Add click to dismiss
        flash.addEventListener('click', () => dismissFlash(flash));
        
        // Auto-dismiss after 5 seconds
        setTimeout(() => {
            if (flash.parentNode) {
                dismissFlash(flash);
            }
        }, 5000);
    });
}

function dismissFlash(flashElement) {
    flashElement.classList.add('fade-out');
    setTimeout(() => {
        if (flashElement.parentNode) {
            flashElement.remove();
        }
    }, 300);
}

function showFlashMessage(message, type = 'info') {
    const flashContainer = document.getElementById('flash-messages');
    if (!flashContainer) return;

    const flash = document.createElement('div');
    flash.className = `flash flash-${type}`;
    flash.textContent = message;
    flash.addEventListener('click', () => dismissFlash(flash));
    
    flashContainer.appendChild(flash);
    
    // Auto-dismiss
    setTimeout(() => {
        if (flash.parentNode) {
            dismissFlash(flash);
        }
    }, 5000);
}

// Modal Management
function initializeModals() {
    // Close modals on overlay click or ESC key
    document.addEventListener('click', function(e) {
        if (e.target.classList.contains('modal-overlay')) {
            closeModal(e.target);
        }
    });

    document.addEventListener('keydown', function(e) {
        if (e.key === 'Escape') {
            const openModal = document.querySelector('.modal-overlay:not(.hidden)');
            if (openModal) {
                closeModal(openModal);
            }
        }
    });

    // Close button handlers
    document.addEventListener('click', function(e) {
        if (e.target.classList.contains('modal-close')) {
            const modal = e.target.closest('.modal-overlay');
            if (modal) {
                closeModal(modal);
            }
        }
    });
}

function openModal(modalId) {
    const modal = document.getElementById(modalId);
    if (modal) {
        modal.classList.remove('hidden');
        modal.setAttribute('aria-hidden', 'false');
        
        // Focus trap
        const focusableElements = modal.querySelectorAll(
            'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
        );
        if (focusableElements.length > 0) {
            focusableElements[0].focus();
        }
    }
}

function closeModal(modal) {
    modal.classList.add('hidden');
    modal.setAttribute('aria-hidden', 'true');
}

// Navigation Management
function initializeNavigation() {
    // Update auth actions based on user session (you could check via API call)
    updateAuthActions();
    
    // Handle mobile menu if needed (currently hidden on mobile)
    // Could add hamburger menu here for mobile navigation
}

function updateAuthActions() {
    // This could make an API call to check authentication status
    // For now, we'll rely on server-side rendering to show correct state
}

// Sample Notes Management
function initializeSampleNotes() {
    const sampleTrigger = document.querySelector('.sample-trigger');
    const sampleDropdown = document.querySelector('.sample-dropdown');
    const patientNoteTextarea = document.getElementById('patient_note');
    
    if (!sampleTrigger || !sampleDropdown || !patientNoteTextarea) return;

    // Sample note data
    const sampleNotes = {
        cardiology: `Patient: 65-year-old male from Austin, Texas
Chief complaint: Chest pain and shortness of breath for 3 days
History: Hypertension, diabetes, family history of CAD
Presentation: Substernal chest pressure, 7/10 severity, radiates to left arm, associated with diaphoresis and nausea. Occurs with minimal exertion.
Duration: 3 days, worsening
Physical: Elevated BP 160/95, heart rate 95, mild bilateral lower extremity edema`,

        neurology: `Patient: 42-year-old female from Portland, Oregon  
Chief complaint: Severe headaches with visual changes for 2 weeks
History: Migraines since age 20, recent increase in frequency
Presentation: Throbbing bilateral headache, 8/10 severity, photophobia, phonophobia, visual aura with zigzag patterns lasting 20-30 minutes
Duration: 2 weeks, daily episodes
Associated symptoms: Nausea, vomiting, difficulty concentrating`,

        gastro: `Patient: 38-year-old female from Miami, Florida
Chief complaint: Abdominal pain and altered bowel habits for 6 weeks  
History: No significant past medical history
Presentation: Crampy lower abdominal pain, alternating constipation and diarrhea, bloating, mucus in stool
Duration: 6 weeks, progressively worsening
Associated symptoms: Weight loss of 8 lbs, occasional blood in stool, fatigue`
    };

    // Toggle dropdown
    sampleTrigger.addEventListener('click', function(e) {
        e.preventDefault();
        sampleDropdown.classList.toggle('hidden');
    });

    // Handle sample selection
    sampleDropdown.addEventListener('click', function(e) {
        if (e.target.classList.contains('sample-option')) {
            const sampleType = e.target.dataset.sample;
            if (sampleNotes[sampleType]) {
                patientNoteTextarea.value = sampleNotes[sampleType];
                updateCharCounter(patientNoteTextarea);
                sampleDropdown.classList.add('hidden');
            }
        }
    });

    // Close dropdown when clicking outside
    document.addEventListener('click', function(e) {
        if (!sampleTrigger.contains(e.target) && !sampleDropdown.contains(e.target)) {
            sampleDropdown.classList.add('hidden');
        }
    });
}

// Copy to Clipboard
function initializeCopyButtons() {
    document.addEventListener('click', function(e) {
        if (e.target.closest('.copy-btn')) {
            const button = e.target.closest('.copy-btn');
            const textToCopy = button.dataset.copyText;
            
            if (textToCopy) {
                copyToClipboard(textToCopy, button);
            }
        }
    });
}

async function copyToClipboard(text, button) {
    try {
        await navigator.clipboard.writeText(text);
        
        // Visual feedback
        const originalText = button.innerHTML;
        button.innerHTML = `
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M20 6L9 17L4 12" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
            </svg>
            Copied!
        `;
        button.classList.add('btn-success');
        
        setTimeout(() => {
            button.innerHTML = originalText;
            button.classList.remove('btn-success');
        }, 2000);
        
    } catch (err) {
        console.error('Failed to copy text: ', err);
        showFlashMessage('Failed to copy text', 'error');
    }
}

// Character Counter
function updateCharCounter(textarea) {
    const counter = document.getElementById('note-counter');
    if (counter) {
        const length = textarea.value.length;
        const maxLength = textarea.getAttribute('maxlength') || 2000;
        counter.textContent = `${length} / ${maxLength}`;
        
        // Color coding
        const percentage = length / maxLength;
        if (percentage > 0.9) {
            counter.style.color = 'var(--error-500)';
        } else if (percentage > 0.75) {
            counter.style.color = 'var(--warning-500)';
        } else {
            counter.style.color = 'var(--neutral-500)';
        }
    }
}

// Progress Animations
function initializeProgressAnimations() {
    // Animate progress rings
    const progressRings = document.querySelectorAll('.progress-ring-fill');
    progressRings.forEach(ring => {
        const percentage = parseFloat(ring.dataset.percentage) || 0;
        const circumference = 2 * Math.PI * 40; // radius = 40
        const strokeDasharray = (percentage / 100) * circumference;
        
        // Animate from 0 to target value
        setTimeout(() => {
            ring.style.strokeDasharray = `${strokeDasharray} ${circumference}`;
        }, 100);
    });

    // Animate progress bars
    const progressBars = document.querySelectorAll('.progress-fill[data-width]');
    progressBars.forEach((bar, index) => {
        const width = parseFloat(bar.dataset.width) || 0;
        
        // Stagger animation
        setTimeout(() => {
            bar.style.width = `${width}%`;
        }, 100 + (index * 50));
    });
}

// Utility Functions
function debounce(func, wait, immediate) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            timeout = null;
            if (!immediate) func(...args);
        };
        const callNow = immediate && !timeout;
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
        if (callNow) func(...args);
    };
}

function showLoading(element) {
    element.classList.add('loading');
    const text = element.querySelector('.btn-text');
    const loading = element.querySelector('.btn-loading');
    if (text) text.style.display = 'none';
    if (loading) loading.style.display = 'flex';
}

function hideLoading(element) {
    element.classList.remove('loading');
    const text = element.querySelector('.btn-text');
    const loading = element.querySelector('.btn-loading');
    if (text) text.style.display = '';
    if (loading) loading.style.display = 'none';
}

// Smooth scroll for anchor links
document.addEventListener('click', function(e) {
    if (e.target.matches('a[href^="#"]')) {
        e.preventDefault();
        const target = document.querySelector(e.target.getAttribute('href'));
        if (target) {
            target.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    }
});

// Enhanced accessibility: Skip links
function createSkipLink() {
    const skipLink = document.createElement('a');
    skipLink.href = '#main-content';
    skipLink.className = 'sr-only';
    skipLink.textContent = 'Skip to main content';
    skipLink.addEventListener('focus', function() {
        this.classList.remove('sr-only');
    });
    skipLink.addEventListener('blur', function() {
        this.classList.add('sr-only');
    });
    document.body.insertBefore(skipLink, document.body.firstChild);
}

// Initialize skip link
createSkipLink();

// Export for use in other scripts
window.MedReferral = {
    showFlashMessage,
    openModal,
    closeModal,
    showLoading,
    hideLoading,
    debounce,
    updateCharCounter
};