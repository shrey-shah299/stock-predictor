// Initialize Lucide icons
document.addEventListener('DOMContentLoaded', function() {
    lucide.createIcons();
    
    // Mobile menu functionality
    const mobileMenuBtn = document.getElementById('mobile-menu-btn');
    const mobileMenu = document.getElementById('mobile-menu');
    const menuIcon = mobileMenuBtn.querySelector('[data-lucide="menu"]');
    const closeIcon = mobileMenuBtn.querySelector('[data-lucide="x"]');
    
    let isMenuOpen = false;
    
    mobileMenuBtn.addEventListener('click', function() {
        isMenuOpen = !isMenuOpen;
        
        if (isMenuOpen) {
            mobileMenu.classList.remove('hidden');
            mobileMenu.classList.add('animate-fade-in');
            // Change icon to X
            mobileMenuBtn.innerHTML = '<i data-lucide="x" class="h-6 w-6"></i>';
        } else {
            mobileMenu.classList.add('hidden');
            mobileMenu.classList.remove('animate-fade-in');
            // Change icon back to menu
            mobileMenuBtn.innerHTML = '<i data-lucide="menu" class="h-6 w-6"></i>';
        }
        
        // Re-initialize icons after changing
        lucide.createIcons();
    });
    
    // Close mobile menu when clicking on links
    const mobileLinks = mobileMenu.querySelectorAll('a');
    mobileLinks.forEach(link => {
        link.addEventListener('click', function() {
            isMenuOpen = false;
            mobileMenu.classList.add('hidden');
            mobileMenu.classList.remove('animate-fade-in');
            mobileMenuBtn.innerHTML = '<i data-lucide="menu" class="h-6 w-6"></i>';
            lucide.createIcons();
        });
    });
    
    // Smooth scrolling for anchor links
    const anchorLinks = document.querySelectorAll('a[href^="#"]');
    anchorLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            const targetId = this.getAttribute('href').substring(1);
            const targetElement = document.getElementById(targetId);
            
            if (targetElement) {
                const headerHeight = 80; // Account for fixed header
                const targetPosition = targetElement.offsetTop - headerHeight;
                
                window.scrollTo({
                    top: targetPosition,
                    behavior: 'smooth'
                });
            }
        });
    });
    
    // Stock search functionality
    const searchInput = document.querySelector('input[type="text"]');
    const predictButton = searchInput.nextElementSibling;
    
    predictButton.addEventListener('click', function() {
        const symbol = searchInput.value.trim().toUpperCase();
        if (symbol) {
            alert(`Generating prediction for ${symbol}...`);
            // Here you would normally make an API call to get real predictions
        } else {
            alert('Please enter a stock symbol');
        }
    });
    
    // Quick prediction buttons
    const quickButtons = document.querySelectorAll('button[class*="px-3 py-1"]');
    quickButtons.forEach(button => {
        if (button.textContent.match(/^[A-Z]{3,5}$/)) {
            button.addEventListener('click', function() {
                const symbol = this.textContent;
                searchInput.value = symbol;
                alert(`Generating prediction for ${symbol}...`);
            });
        }
    });
    
    // Add some interactive animations
    const cards = document.querySelectorAll('.bg-card');
    cards.forEach(card => {
        card.addEventListener('mouseenter', function() {
            this.style.transform = 'translateY(-2px)';
            this.style.transition = 'transform 0.2s ease';
        });
        
        card.addEventListener('mouseleave', function() {
            this.style.transform = 'translateY(0)';
        });
    });
    
    // Simulate live updates for prediction cards
    function updatePredictions() {
        const predictionCards = document.querySelectorAll('#predictions .bg-card');
        predictionCards.forEach(card => {
            const currentPrice = card.querySelector('.space-y-2 .flex:first-child span:last-child');
            if (currentPrice && currentPrice.textContent.includes('$')) {
                // Simulate small price changes
                const price = parseFloat(currentPrice.textContent.replace('$', ''));
                const change = (Math.random() - 0.5) * 2; // Random change between -1 and 1
                const newPrice = (price + change).toFixed(2);
                currentPrice.textContent = `$${newPrice}`;
            }
        });
    }
    
    // Update predictions every 5 seconds
    setInterval(updatePredictions, 5000);
});

// Add some CSS animations
const style = document.createElement('style');
style.textContent = `
    @keyframes fade-in {
        from {
            opacity: 0;
            transform: translateY(-10px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .animate-fade-in {
        animation: fade-in 0.3s ease-out;
    }
    
    .animate-pulse {
        animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
    }
    
    @keyframes pulse {
        0%, 100% {
            opacity: 1;
        }
        50% {
            opacity: 0.5;
        }
    }
`;
document.head.appendChild(style);