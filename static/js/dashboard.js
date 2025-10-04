const initialiseCharts = () => {
    if (typeof Plotly === 'undefined') {
        return;
    }

    const chartsToRender = [];

    const multiChartScript = document.getElementById('chart-configs-data');
    if (multiChartScript) {
        try {
            const payload = JSON.parse(multiChartScript.textContent || '[]');
            payload.forEach((entry) => {
                if (!entry || !entry.id || !entry.config) {
                    return;
                }
                const target = document.getElementById(entry.id);
                if (!target) {
                    return;
                }
                chartsToRender.push({ element: target, config: entry.config });
            });
        } catch (error) {
            console.error('Unable to parse chart configuration payload', error);
        }
    }

    const singleChartScript = document.getElementById('single-chart-config');
    if (singleChartScript) {
        try {
            const chartConfig = JSON.parse(singleChartScript.textContent || '{}');
            const target = document.getElementById('trades-chart');
            if (target && chartConfig && chartConfig.data && chartConfig.layout) {
                chartsToRender.push({ element: target, config: chartConfig });
            }
        } catch (error) {
            console.error('Unable to parse single chart configuration', error);
        }
    }

    chartsToRender.forEach(({ element, config }) => {
        const layout = { ...(config.layout || {}), autosize: true };
        const chart = Plotly.newPlot(
            element,
            config.data || [],
            layout,
            { responsive: true, displayModeBar: false }
        );
        if (chart) {
            window.addEventListener('resize', () => {
                Plotly.Plots.resize(element);
            });
        }
    });
};

document.addEventListener('DOMContentLoaded', initialiseCharts);
