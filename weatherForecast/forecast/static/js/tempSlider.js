// Set up the temperature slider based on data attributes
document.addEventListener('DOMContentLoaded', function () {
    // Get the progress bar and thumb elements
    const progressBar = document.querySelector('.temp-slider-progress');
    const sliderThumb = document.querySelector('.temp-slider-thumb');

    if (progressBar && sliderThumb) {
        // Get percentage values from data attributes
        const percentage = progressBar.dataset.percentage;

        // Apply styles directly
        progressBar.style.width = percentage + '%';
        sliderThumb.style.left = percentage + '%';
    }
});

// Set mini slider progress widths based on data-value attribute
document.addEventListener('DOMContentLoaded', function () {
    const miniSliders = document.querySelectorAll('.mini-slider-progress');

    miniSliders.forEach(slider => {
        const value = slider.getAttribute('data-value');
        if (value) {
            slider.style.width = value + '%';
        }
    });
}); 