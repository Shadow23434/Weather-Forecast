document.addEventListener('DOMContentLoaded', function () {
    const cityInput = document.getElementById('cityInput');
    const searchBtn = document.getElementById('searchBtn');
    const closeBtn = document.getElementById('closeBtn');
    const searchForm = document.getElementById('searchForm');

    // Function to toggle buttons visibility
    function toggleButtons() {
        if (cityInput.value.trim() !== '') {
            searchBtn.style.display = 'none';
            closeBtn.style.display = 'block';
        } else {
            searchBtn.style.display = 'block';
            closeBtn.style.display = 'none';
        }
    }

    // Initial check
    toggleButtons();

    // Listen for input changes
    cityInput.addEventListener('input', toggleButtons);

    // Handle close button click
    closeBtn.addEventListener('click', function () {
        cityInput.value = '';
        toggleButtons();
        cityInput.focus();
    });

    // Handle form submission
    searchForm.addEventListener('submit', function (e) {
        if (cityInput.value.trim() === '') {
            e.preventDefault();
        }
    });
}); 