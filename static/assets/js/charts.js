// Charts JavaScript - Progress rings, bars, and data visualization

document.addEventListener('DOMContentLoaded', function() {
    initializeProgressRings();
    initializeProgressBars();
    initializeDistributionCharts();
});

// Progress Ring Animations
function initializeProgressRings() {
    const progressRings = document.querySelectorAll('.progress-ring-fill');
    
    progressRings.forEach((ring, index) => {
        const percentage = parseFloat(ring.dataset.percentage) || 0;
        animateProgressRing(ring, percentage, index * 100);
    });
}

function animateProgressRing(ringElement, percentage, delay = 0) {
    const radius = 40; // Based on our SVG radius
    const circumference = 2 * Math.PI * radius;
    const strokeDasharray = (percentage / 100) * circumference;
    
    // Set initial state
    ringElement.style.strokeDasharray = `0 ${circumference}`;
    
    // Animate to target
    setTimeout(() => {
        ringElement.style.strokeDasharray = `${strokeDasharray} ${circumference}`;
        
        // Update percentage text with counting animation
        const progressText = ringElement.closest('.progress-ring-container')?.querySelector('.progress-percentage');
        if (progressText) {
            animateNumber(progressText, 0, percentage, 1000, '%');
        }
    }, delay);
}

// Progress Bar Animations
function initializeProgressBars() {
    const progressBars = document.querySelectorAll('.progress-fill[data-width]');
    
    progressBars.forEach((bar, index) => {
        const width = parseFloat(bar.dataset.width) || 0;
        animateProgressBar(bar, width, index * 50);
    });
}

function animateProgressBar(barElement, width, delay = 0) {
    // Set initial state
    barElement.style.width = '0%';
    
    // Animate to target
    setTimeout(() => {
        barElement.style.width = `${width}%`;
    }, delay);
}

// Distribution Chart Interactions
function initializeDistributionCharts() {
    const distributionItems = document.querySelectorAll('.distribution-item');
    
    distributionItems.forEach(item => {
        // Enhance keyboard navigation
        item.addEventListener('keydown', function(e) {
            if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                const link = this.querySelector('.distribution-link');
                if (link) {
                    link.click();
                }
            }
        });
        
        // Add hover animations
        item.addEventListener('mouseenter', function() {
            const progressBar = this.querySelector('.progress-fill');
            if (progressBar) {
                progressBar.style.transform = 'scaleY(1.1)';
                progressBar.style.transformOrigin = 'bottom';
            }
        });
        
        item.addEventListener('mouseleave', function() {
            const progressBar = this.querySelector('.progress-fill');
            if (progressBar) {
                progressBar.style.transform = 'scaleY(1)';
            }
        });
    });
}

// Number Animation Utility
function animateNumber(element, start, end, duration, suffix = '') {
    const startTime = performance.now();
    const difference = end - start;
    
    function updateNumber(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);
        
        // Easing function (ease-out)
        const easedProgress = 1 - Math.pow(1 - progress, 3);
        
        const current = Math.round(start + (difference * easedProgress));
        element.textContent = current + suffix;
        
        if (progress < 1) {
            requestAnimationFrame(updateNumber);
        }
    }
    
    requestAnimationFrame(updateNumber);
}

// Create Progress Ring SVG programmatically
function createProgressRing(percentage, size = 80) {
    const radius = (size - 16) / 2; // Account for stroke width
    const circumference = 2 * Math.PI * radius;
    const strokeDasharray = (percentage / 100) * circumference;
    
    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.setAttribute('class', 'progress-ring');
    svg.setAttribute('width', size);
    svg.setAttribute('height', size);
    svg.setAttribute('viewBox', `0 0 ${size} ${size}`);
    svg.style.transform = 'rotate(-90deg)';
    
    // Background circle
    const bgCircle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
    bgCircle.setAttribute('class', 'progress-ring-circle');
    bgCircle.setAttribute('cx', size / 2);
    bgCircle.setAttribute('cy', size / 2);
    bgCircle.setAttribute('r', radius);
    
    // Progress circle
    const progressCircle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
    progressCircle.setAttribute('class', 'progress-ring-fill');
    progressCircle.setAttribute('cx', size / 2);
    progressCircle.setAttribute('cy', size / 2);
    progressCircle.setAttribute('r', radius);
    progressCircle.style.strokeDasharray = `0 ${circumference}`;
    
    svg.appendChild(bgCircle);
    svg.appendChild(progressCircle);
    
    // Animate
    setTimeout(() => {
        progressCircle.style.strokeDasharray = `${strokeDasharray} ${circumference}`;
    }, 100);
    
    return svg;
}

// Staggered Animation for Lists
function staggerAnimation(elements, delay = 100) {
    elements.forEach((element, index) => {
        element.style.opacity = '0';
        element.style.transform = 'translateY(20px)';
        
        setTimeout(() => {
            element.style.transition = 'all 0.5s ease-out';
            element.style.opacity = '1';
            element.style.transform = 'translateY(0)';
        }, index * delay);
    });
}

// Initialize staggered animations for existing elements
document.addEventListener('DOMContentLoaded', function() {
    // Animate feature cards on landing page
    const featureCards = document.querySelectorAll('.feature-card');
    if (featureCards.length > 0) {
        setTimeout(() => staggerAnimation(featureCards, 150), 500);
    }
    
    // Animate clinic cards
    const clinicCards = document.querySelectorAll('.clinic-card');
    if (clinicCards.length > 0) {
        setTimeout(() => staggerAnimation(clinicCards, 100), 300);
    }
});

// Chart color utilities
function getChartColor(index, total) {
    const colors = [
        'var(--secondary-600)',
        'var(--secondary-600)', 
        'var(--warning-500)',
        'var(--neutral-600)',
        'var(--primary-400)',
        'var(--secondary-400)'
    ];
    
    return colors[index % colors.length];
}

// Export for use in other scripts
window.ChartUtils = {
    animateProgressRing,
    animateProgressBar,
    animateNumber,
    createProgressRing,
    staggerAnimation,
    getChartColor
};