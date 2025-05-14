/**
 * Weather Icons handling
 * Provides fallback and appropriate handling for weather icon display
 */

document.addEventListener('DOMContentLoaded', function () {
    // Get weather icon element
    const weatherIcon = document.getElementById('weatherIcon');

    if (weatherIcon) {
        // Handle error loading the icon
        weatherIcon.addEventListener('error', function () {
            // Try to extract the weather condition
            const description = weatherIcon.alt.toLowerCase();
            let iconName = 'clear-day';

            // Determine appropriate fallback based on weather description
            if (description.includes('rain') || description.includes('shower') || description.includes('drizzle')) {
                iconName = 'rain';
            } else if (description.includes('cloud')) {
                iconName = 'cloudy';
            } else if (description.includes('overcast')) {
                iconName = 'overcast';
            } else if (description.includes('mist') || description.includes('haze') || description.includes('fog')) {
                iconName = 'mist';
            } else if (description.includes('snow') || description.includes('blizzard')) {
                iconName = 'snow';
            } else if (description.includes('sleet')) {
                iconName = 'sleet';
            } else if (description.includes('thunder') || description.includes('storm')) {
                iconName = 'thunderstorm';
            } else if (description.includes('clear') || description.includes('sunny')) {
                iconName = 'clear-day';
            }

            // Update the icon source
            const iconPath = `/static/img/icons/${iconName}.svg`;
            weatherIcon.src = iconPath;

            console.log(`Applied fallback weather icon: ${iconName} for "${description}"`);
        });
    }
}); 