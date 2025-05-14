document.addEventListener('DOMContentLoaded', () => {
    console.log('Chart setup started');
    const chartElement = document.getElementById('chart');
    if (!chartElement) {
        console.error('Canvas Element not found!')
        return;
    }
    console.log('Canvas element found');

    const ctx = chartElement.getContext('2d');

    const forecastItems = document.querySelectorAll('.forecast-item');
    console.log('Found forecast items:', forecastItems.length);

    if (forecastItems.length === 0) {
        console.error('No forecast items found in the DOM');
        return;
    }

    const temps = [];
    const times = [];
    const forecastItemsArray = Array.from(forecastItems);

    forecastItemsArray.forEach((item, index) => {
        const timeElement = item.querySelector('.forecast-time');
        const tempElement = item.querySelector('.forecast-temperatureValue');

        if (!timeElement || !tempElement) {
            console.error(`Missing elements in forecast item ${index}:`, {
                timeElement: !!timeElement,
                tempElement: !!tempElement
            });
            return;
        }

        const time = timeElement.textContent;
        const temp = tempElement.textContent;

        if (time && temp) {
            times.push(time);
            const tempValue = parseFloat(temp);
            if (!isNaN(tempValue)) {
                temps.push(tempValue);
            }
        }
    });

    // Ensure all values are valid before using them
    if (temps.length === 0 || times.length === 0) {
        console.error('Temp or time values are missing!');
        return;
    }

    try {
        // Destroy existing chart if it exists
        if (window.temperatureChart) {
            window.temperatureChart.destroy();
        }

        // Create new chart
        window.temperatureChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: times,
                datasets: [
                    {
                        data: temps,
                        borderColor: '#2DBEBE', // Turquoise color
                        backgroundColor: 'transparent',
                        borderWidth: 3,
                        tension: 0.4, // Make the line curved
                        fill: false,
                        pointRadius: 5, // Show small points
                        pointBackgroundColor: '#2DBEBE', // Match line color
                        pointBorderColor: 'white',
                        pointBorderWidth: 1,
                        pointHoverRadius: 6
                    },
                ],
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        enabled: true,
                        backgroundColor: 'rgba(0, 0, 0, 0.7)',
                        titleFont: { size: 12 },
                        bodyFont: { size: 12 },
                        padding: 8,
                        displayColors: false,
                        callbacks: {
                            label: function (context) {
                                return `${context.parsed.y}°`;
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        display: true,
                        grid: {
                            display: false
                        },
                        ticks: {
                            color: 'rgba(255, 255, 255, 0.7)',
                            font: {
                                size: 0 // Hide text but keep positions
                            },
                            maxRotation: 0,
                            minRotation: 0
                        },
                        afterTickToLabelConversion: function (scaleInstance) {
                            // This sets all labels to blank to hide them but keep spacing
                            scaleInstance.ticks.forEach(function (tick) {
                                tick.label = '';
                            });
                        }
                    },
                    y: {
                        display: false,
                        grid: {
                            display: false
                        },
                        min: Math.min(...temps) - 3,
                        max: Math.max(...temps) + 3
                    },
                },
                animation: {
                    duration: 1000
                },
                layout: {
                    padding: {
                        top: 15,
                        right: 15,
                        bottom: 15,
                        left: 15
                    }
                }
            },
            plugins: [{
                id: 'alignForecastItems',
                afterRender: (chart) => {
                    // Get the horizontal positions of each point in the chart
                    const chartPoints = chart.getDatasetMeta(0).data;
                    const forecastContainer = document.querySelector('.forecast');

                    if (!forecastContainer || chartPoints.length !== forecastItemsArray.length) {
                        return;
                    }

                    // Adjust chart container width to match the forecast container
                    const chartContainer = document.querySelector('.chart-container');
                    if (chartContainer) {
                        chartContainer.style.width = forecastContainer.offsetWidth + 'px';
                        chartContainer.style.marginLeft = 'auto';
                        chartContainer.style.marginRight = 'auto';
                    }
                }
            }]
        });
        console.log('Chart created successfully');
    } catch (error) {
        console.error('Error creating chart:', error);
    }
});