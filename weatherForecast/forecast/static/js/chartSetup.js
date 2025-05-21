// Wait for DOM to be fully loaded
document.addEventListener('DOMContentLoaded', () => {
    console.log('Chart setup started');

    // Make sure Chart.js is loaded
    if (typeof Chart === 'undefined') {
        console.error('Chart.js not loaded! Please check the script inclusion.');

        // Try to load Chart.js dynamically if it's not available
        const chartScript = document.createElement('script');
        chartScript.src = 'https://cdn.jsdelivr.net/npm/chart.js@3.9.1/dist/chart.min.js';
        chartScript.onload = initChart;
        document.head.appendChild(chartScript);
        return;
    }

    // Use setTimeout to ensure DOM is fully rendered
    setTimeout(initChart, 500);
});

// Separate function to initialize the chart
function initChart() {
    console.log('Initializing chart...');

    const chartElement = document.getElementById('chart');
    if (!chartElement) {
        console.error('Canvas Element not found!');
        return;
    }
    console.log('Canvas element found');

    // Set explicit dimensions on the canvas element
    chartElement.style.height = '270px';
    chartElement.height = 270; // Set the height attribute
    chartElement.style.width = '100%';

    const chartContainer = document.querySelector('.chart-container');
    if (chartContainer) {
        // Ensure container is visible with proper dimensions
        chartContainer.style.display = 'block';
        chartContainer.style.height = '300px';
        chartContainer.style.minHeight = '300px';
    }

    // Check if canvas is visible and has dimensions
    const chartWidth = chartElement.offsetWidth;
    const chartHeight = chartElement.offsetHeight || 270; // Use default if 0
    console.log('Chart dimensions:', chartWidth, 'x', chartHeight);

    const ctx = chartElement.getContext('2d');

    const forecastItems = document.querySelectorAll('.forecast-item');
    console.log('Found forecast items:', forecastItems.length);

    if (forecastItems.length === 0) {
        console.error('No forecast items found in the DOM');
        return;
    }

    const avgTemps = [];
    const dates = [];
    const forecastItemsArray = Array.from(forecastItems);

    forecastItemsArray.forEach((item, index) => {
        const dateElement = item.querySelector('.forecast-date');
        const minTempElement = item.querySelector('.forecast-min-temp .forecast-temp-value');
        const maxTempElement = item.querySelector('.forecast-max-temp .forecast-temp-value');

        if (!dateElement || !minTempElement || !maxTempElement) {
            console.error(`Missing elements in forecast item ${index}:`, {
                dateElement: !!dateElement,
                minTempElement: !!minTempElement,
                maxTempElement: !!maxTempElement
            });
            return;
        }

        const date = dateElement.textContent;
        const minTemp = parseFloat(minTempElement.textContent);
        const maxTemp = parseFloat(maxTempElement.textContent);

        if (date && !isNaN(minTemp) && !isNaN(maxTemp)) {
            dates.push(date);

            // Calculate average temperature for the day
            const avgTemp = (minTemp + maxTemp) / 2;
            avgTemps.push(parseFloat(avgTemp.toFixed(1)));
        }
    });

    // Ensure all values are valid before using them
    if (avgTemps.length === 0 || dates.length === 0) {
        console.error('Temperature or date values are missing!');
        return;
    }

    console.log('Chart data prepared:', {
        dates: dates,
        avgTemps: avgTemps
    });

    try {
        // Destroy existing chart if it exists
        if (window.temperatureChart) {
            window.temperatureChart.destroy();
        }

        // Disable animations to prevent storage access errors
        Chart.defaults.animation = false;

        // Create simplified chart with only average temperature
        window.temperatureChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: dates,
                datasets: [
                    {
                        label: 'Average Temperature',
                        data: avgTemps,
                        borderColor: '#2DBEBE', // Turquoise color
                        backgroundColor: 'rgba(45, 190, 190, 0.1)',
                        borderWidth: 3,
                        tension: 0.4, // Make the line curved
                        fill: true,
                        pointRadius: 6, // Larger points for better visibility
                        pointBackgroundColor: '#2DBEBE', // Match line color
                        pointBorderColor: 'white',
                        pointBorderWidth: 2,
                        pointHoverRadius: 8
                    }
                ],
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: false, // Disable animations to prevent storage access issues
                plugins: {
                    legend: {
                        display: true,
                        position: 'top',
                        labels: {
                            color: 'rgba(255, 255, 255, 0.7)',
                            boxWidth: 15,
                            padding: 10,
                            font: {
                                size: 12,
                                weight: 'bold'
                            }
                        }
                    },
                    tooltip: {
                        enabled: true,
                        backgroundColor: 'rgba(0, 0, 0, 0.7)',
                        titleFont: { size: 12 },
                        bodyFont: { size: 12 },
                        padding: 8,
                        displayColors: true,
                        callbacks: {
                            label: function (context) {
                                return `${context.dataset.label}: ${context.parsed.y}°C`;
                            },
                            title: function (tooltipItems) {
                                return `${tooltipItems[0].label}`;
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        display: true,
                        grid: {
                            display: false,
                            color: 'rgba(255, 255, 255, 0.1)'
                        },
                        ticks: {
                            color: 'rgba(255, 255, 255, 0.7)',
                            font: {
                                size: 11
                            },
                            maxRotation: 0,
                            minRotation: 0
                        }
                    },
                    y: {
                        display: true,
                        grid: {
                            display: true,
                            color: 'rgba(255, 255, 255, 0.1)'
                        },
                        ticks: {
                            color: 'rgba(255, 255, 255, 0.7)',
                            font: {
                                size: 11
                            },
                            callback: function (value) {
                                return value + '°C';
                            }
                        },
                        min: Math.min(...avgTemps) - 2,
                        max: Math.max(...avgTemps) + 2
                    },
                },
                layout: {
                    padding: {
                        top: 20,
                        right: 20,
                        bottom: 10,
                        left: 10
                    }
                }
            }
        });
        console.log('Chart created successfully');
    } catch (error) {
        console.error('Error creating chart:', error);
    }
}