// Dashboard visualization initialization
class ReferralDashboard {
    constructor() {
        this.data = window.referralData || {};
        this.charts = {};
        this.init();
    }

    init() {
        this.createReferralChart();
        this.createSymptomRadar();
        this.setupInteractions();
        this.animateCards();
    }

    createReferralChart() {
        const ctx = document.getElementById('referralChart');
        if (!ctx || !this.data.distribution) return;

        const labels = Object.keys(this.data.distribution);
        const data = Object.values(this.data.distribution);
        const colors = this.generateColors(labels.length);

        this.charts.referral = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: labels,
                datasets: [{
                    data: data,
                    backgroundColor: colors.backgrounds,
                    borderColor: colors.borders,
                    borderWidth: 2,
                    hoverOffset: 8
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: {
                            padding: 20,
                            usePointStyle: true,
                            font: {
                                size: 12,
                                weight: '500'
                            }
                        }
                    },
                    tooltip: {
                        callbacks: {
                            label: (context) => {
                                return `${context.label}: ${context.parsed}%`;
                            }
                        }
                    }
                },
                cutout: '60%',
                animation: {
                    animateRotate: true,
                    duration: 2000
                }
            }
        });

        // Add center text
        this.addCenterText(ctx, this.data.recommended, `${data[0]}%`);
    }

    createSymptomRadar() {
        const ctx = document.getElementById('symptomRadar');
        if (!ctx || !this.data.extractedData) return;

        const symptoms = [
            'Chest Pain', 'Headache', 'Neurological', 
            'Respiratory', 'GI', 'Fever', 'Bleeding', 'Swelling'
        ];
        
        const intensityData = symptoms.map(symptom => {
            const key = `${symptom.toLowerCase().replace(' ', '_')}_intensity`;
            return this.data.extractedData[key] || 0;
        });

        this.charts.symptom = new Chart(ctx, {
            type: 'radar',
            data: {
                labels: symptoms,
                datasets: [{
                    label: 'Symptom Intensity',
                    data: intensityData,
                    fill: true,
                    backgroundColor: 'rgba(102, 126, 234, 0.2)',
                    borderColor: 'rgb(102, 126, 234)',
                    pointBackgroundColor: 'rgb(102, 126, 234)',
                    pointBorderColor: '#fff',
                    pointHoverBackgroundColor: '#fff',
                    pointHoverBorderColor: 'rgb(102, 126, 234)',
                    borderWidth: 2,
                    pointRadius: 6,
                    pointHoverRadius: 8
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                elements: {
                    line: {
                        borderWidth: 3
                    }
                },
                plugins: {
                    legend: {
                        display: false
                    }
                },
                scales: {
                    r: {
                        angleLines: {
                            display: true,
                            color: 'rgba(0, 0, 0, 0.1)'
                        },
                        suggestedMin: 0,
                        suggestedMax: 10,
                        ticks: {
                            stepSize: 2,
                            font: {
                                size: 10
                            }
                        },
                        pointLabels: {
                            font: {
                                size: 12,
                                weight: '500'
                            }
                        }
                    }
                },
                animation: {
                    duration: 2000,
                    easing: 'easeInOutQuart'
                }
            }
        });
    }

    addCenterText(ctx, title, subtitle) {
        const chart = Chart.getChart(ctx);
        Chart.register({
            id: 'centerText',
            beforeDraw: (chart) => {
                if (chart.canvas.id !== 'referralChart') return;
                
                const { ctx, chartArea: { top, width, height } } = chart;
                const centerX = width / 2;
                const centerY = top + height / 2;

                ctx.save();
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';

                // Title
                ctx.font = 'bold 16px Inter';
                ctx.fillStyle = '#1f2937';
                ctx.fillText(title, centerX, centerY - 10);

                // Subtitle
                ctx.font = '600 24px Inter';
                ctx.fillStyle = '#667eea';
                ctx.fillText(subtitle, centerX, centerY + 15);

                ctx.restore();
            }
        });
    }

    generateColors(count) {
        const baseColors = [
            '#667eea', '#f093fb', '#4facfe', '#43e97b',
            '#fa709a', '#ffecd2', '#a8edea', '#d299c2'
        ];
        
        const backgrounds = [];
        const borders = [];

        for (let i = 0; i < count; i++) {
            const color = baseColors[i % baseColors.length];
            backgrounds.push(color + '20'); // 20% opacity
            borders.push(color);
        }

        return { backgrounds, borders };
    }

    setupInteractions() {
        // Specialist detail buttons
        document.querySelectorAll('.specialist-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const specialist = e.target.dataset.specialist;
                window.location.href = `/specialist/${specialist}`;
            });
        });

        // Share functionality
        document.querySelector('.share-btn')?.addEventListener('click', () => {
            this.shareResults();
        });

        // Chart interactions
        if (this.charts.referral) {
            this.charts.referral.options.onHover = (event, activeElements) => {
                if (activeElements.length > 0) {
                    event.native.target.style.cursor = 'pointer';
                } else {
                    event.native.target.style.cursor = 'default';
                }
            };
        }
    }

    animateCards() {
        const cards = document.querySelectorAll('.dashboard-grid > *');
        const observer = new IntersectionObserver((entries) => {
            entries.forEach((entry, index) => {
                if (entry.isIntersecting) {
                    setTimeout(() => {
                        entry.target.classList.add('animate-in');
                    }, index * 100);
                }
            });
        }, { threshold: 0.1 });

        cards.forEach(card => observer.observe(card));
    }

    shareResults() {
        if (navigator.share) {
            navigator.share({
                title: 'Medical Referral Analysis',
                text: `AI recommends ${this.data.recommended} with ${this.data.distribution[this.data.recommended]}% confidence`,
                url: window.location.href
            });
        } else {
            // Fallback - copy to clipboard
            const shareText = `Medical Referral Analysis\n\nRecommendation: ${this.data.recommended}\nConfidence: ${this.data.distribution[this.data.recommended]}%\n\n${window.location.href}`;
            navigator.clipboard.writeText(shareText).then(() => {
                this.showNotification('Results copied to clipboard!');
            });
        }
    }

    showNotification(message) {
        const notification = document.createElement('div');
        notification.className = 'notification success';
        notification.textContent = message;
        document.body.appendChild(notification);
        
        setTimeout(() => {
            notification.classList.add('show');
        }, 100);
        
        setTimeout(() => {
            notification.classList.remove('show');
            setTimeout(() => notification.remove(), 300);
        }, 3000);
    }
}

// Initialize dashboard when DOM loads
document.addEventListener('DOMContentLoaded', () => {
    new ReferralDashboard();
});
