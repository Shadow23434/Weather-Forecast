// Loading overlay functionality
const loadingOverlay = document.getElementById('loadingOverlay');
const searchForm = document.getElementById('searchForm');

// Show loading overlay
function showLoading() {
    loadingOverlay.classList.add('show');
    document.body.style.overflow = 'hidden'; // Prevent scrolling while loading
}

// Hide loading overlay
function hideLoading() {
    loadingOverlay.classList.remove('show');
    document.body.style.overflow = ''; // Restore scrolling
}

// Handle form submission
searchForm.addEventListener('submit', function (e) {
    // Show loading overlay when form is submitted
    showLoading();

    // The form will continue to submit normally
    // The loading overlay will be visible until the page reloads
});

// Handle page load
document.addEventListener('DOMContentLoaded', function () {
    // Hide loading overlay when page is fully loaded
    hideLoading();
});

// Handle page unload
window.addEventListener('beforeunload', function () {
    // Show loading overlay when navigating away
    showLoading();
}); 