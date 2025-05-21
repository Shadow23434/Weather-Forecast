document.addEventListener('DOMContentLoaded', function () {
    const modal = document.getElementById('errorModal');
    const body = document.querySelector('body');
    const cod = parseInt(body.dataset.cod) || 200;

    console.log(`Modal script loaded. cod=${cod}, modal element:`, modal);

    if (!modal) {
        console.error('Error modal element not found in DOM');
        return;
    }

    if (cod !== 200) {
        console.log('Showing error modal');
        modal.classList.add('show');
    }

    // Attach event listener to the Close button
    const closeButton = modal.querySelector('button[onclick="closeModal()"]');
    if (closeButton) {
        closeButton.addEventListener('click', function () {
            console.log('Close button clicked');
            modal.classList.remove('show');
        });
    } else {
        console.error('Close button not found in modal');
    }
});