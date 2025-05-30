// Function to show loading overlay
function showLoadingOverlay() {
    const overlay = document.getElementById('loadingOverlay');
    if (overlay) {
        overlay.classList.add('show');
    }
}

// Function to hide loading overlay
function hideLoadingOverlay() {
    const overlay = document.getElementById('loadingOverlay');
    if (overlay) {
        overlay.classList.remove('show');
    }
}

// Add event listener to search form
document.addEventListener('DOMContentLoaded', function () {
    const searchForm = document.getElementById('searchForm');
    if (searchForm) {
        searchForm.addEventListener('submit', function () {
            showLoadingOverlay();
        });
    }
});

// Export functions for use in other modules
export { showLoadingOverlay, hideLoadingOverlay }; 