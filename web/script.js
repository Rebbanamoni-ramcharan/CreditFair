/* =========================================================
   CREDITFAIR JAVASCRIPT
========================================================= */


/* =========================================================
   CHART DEFAULTS
========================================================= */

Chart.defaults.font.family =
    "Inter, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif";

Chart.defaults.color = "#98a2b3";

Chart.defaults.plugins.legend.labels.usePointStyle = true;


/* =========================================================
   1. LOAN STATUS CHART
========================================================= */

const loanStatusCanvas =
    document.getElementById("loanStatusChart");

if (loanStatusCanvas) {

    new Chart(
        loanStatusCanvas,
        {
            type: "doughnut",

            data: {

                labels: [
                    "Fully Paid",
                    "Charged Off"
                ],

                datasets: [
                    {
                        data: [
                            23026,
                            3977
                        ],

                        backgroundColor: [
                            "#818cf8",
                            "#fb7185"
                        ],

                        borderWidth: 0,

                        hoverOffset: 8
                    }
                ]
            },

            options: {

                responsive: true,

                maintainAspectRatio: false,

                cutout: "70%",

                plugins: {

                    legend: {
                        position: "bottom",

                        labels: {
                            padding: 20
                        }
                    },

                    tooltip: {

                        callbacks: {

                            label: function(context) {

                                const value =
                                    context.raw;

                                const total =
                                    23026 + 3977;

                                const percentage =
                                    ((value / total) * 100)
                                    .toFixed(2);

                                return (
                                    context.label +
                                    ": " +
                                    value.toLocaleString() +
                                    " (" +
                                    percentage +
                                    "%)"
                                );
                            }
                        }
                    }
                }
            }
        }
    );
}


/* =========================================================
   2. MODEL PERFORMANCE CHART
========================================================= */

const performanceCanvas =
    document.getElementById("performanceChart");

if (performanceCanvas) {

    new Chart(
        performanceCanvas,
        {
            type: "bar",

            data: {

                labels: [
                    "Accuracy",
                    "Precision",
                    "Recall",
                    "F1 Score",
                    "ROC-AUC"
                ],

                datasets: [

                    {
                        label: "Score (%)",

                        data: [
                            65.97,
                            24.72,
                            64.15,
                            35.69,
                            71.25
                        ],

                        backgroundColor: [
                            "#818cf8",
                            "#a78bfa",
                            "#60a5fa",
                            "#38bdf8",
                            "#34d399"
                        ],

                        borderRadius: 7,

                        borderSkipped: false
                    }

                ]
            },

            options: {

                responsive: true,

                maintainAspectRatio: false,

                scales: {

                    y: {

                        beginAtZero: true,

                        max: 100,

                        grid: {
                            color: "#263244"
                        },

                        ticks: {
                            callback: function(value) {
                                return value + "%";
                            }
                        }
                    },

                    x: {

                        grid: {
                            display: false
                        }
                    }
                },

                plugins: {

                    legend: {
                        display: false
                    },

                    tooltip: {

                        callbacks: {

                            label: function(context) {

                                return (
                                    context.raw +
                                    "%"
                                );
                            }
                        }
                    }
                }
            }
        }
    );
}


/* =========================================================
   3. FAIRNESS CHART
========================================================= */

const fairnessCanvas =
    document.getElementById("fairnessChart");

if (fairnessCanvas) {

    new Chart(
        fairnessCanvas,
        {
            type: "bar",

            data: {

                labels: [
                    "MORTGAGE",
                    "OWN",
                    "RENT",
                    "OTHER"
                ],

                datasets: [

                    {
                        label: "Accuracy",

                        data: [
                            66.88,
                            66.82,
                            65.04,
                            53.85
                        ],

                        backgroundColor: "#818cf8",

                        borderRadius: 6
                    },

                    {
                        label: "Recall",

                        data: [
                            60.95,
                            63.24,
                            67.01,
                            100
                        ],

                        backgroundColor: "#34d399",

                        borderRadius: 6
                    }

                ]
            },

            options: {

                responsive: true,

                maintainAspectRatio: false,

                scales: {

                    y: {

                        beginAtZero: true,

                        max: 100,

                        grid: {
                            color: "#263244"
                        },

                        ticks: {

                            callback: function(value) {
                                return value + "%";
                            }
                        }
                    },

                    x: {

                        grid: {
                            display: false
                        }
                    }
                },

                plugins: {

                    legend: {

                        position: "bottom",

                        labels: {
                            padding: 18
                        }
                    },

                    tooltip: {

                        callbacks: {

                            label: function(context) {

                                return (
                                    context.dataset.label +
                                    ": " +
                                    context.raw +
                                    "%"
                                );
                            }
                        }
                    }
                }
            }
        }
    );
}


/* =========================================================
   PREDICTION CHART
========================================================= */

let predictionChart = null;


function createPredictionChart(
    fullyPaidProbability,
    chargedOffProbability
) {

    const canvas =
        document.getElementById("predictionChart");

    if (!canvas) {
        return;
    }


    if (predictionChart) {
        predictionChart.destroy();
    }


    predictionChart = new Chart(
        canvas,
        {
            type: "doughnut",

            data: {

                labels: [
                    "Fully Paid",
                    "Charged Off"
                ],

                datasets: [
                    {
                        data: [
                            fullyPaidProbability,
                            chargedOffProbability
                        ],

                        backgroundColor: [
                            "#818cf8",
                            "#fb7185"
                        ],

                        borderWidth: 0
                    }
                ]
            },

            options: {

                responsive: true,

                maintainAspectRatio: false,

                cutout: "73%",

                plugins: {

                    legend: {
                        display: false
                    },

                    tooltip: {

                        callbacks: {

                            label: function(context) {

                                return (
                                    context.label +
                                    ": " +
                                    context.raw.toFixed(2) +
                                    "%"
                                );
                            }
                        }
                    }
                }
            }
        }
    );
}


/* =========================================================
   INITIAL DEMONSTRATION
========================================================= */

createPredictionChart(
    66.41,
    33.59
);


/* =========================================================
   PREDICTION FUNCTION  (calls real FastAPI backend)
========================================================= */

async function predictLoan() {

    const loanAmount      = parseFloat(document.getElementById("loanAmount").value) || 0;
    const interestRate    = parseFloat(document.getElementById("interestRate").value) || 0;
    const annualIncome    = parseFloat(document.getElementById("annualIncome").value) || 0;
    const homeOwnership   = document.getElementById("homeOwnership").value;
    const purpose         = document.getElementById("purpose").value;
    const dti             = parseFloat(document.getElementById("dti").value) || 0;

    const statusEl      = document.getElementById("predictionStatus");
    const probEl        = document.getElementById("predictionProbability");
    const msgEl         = document.getElementById("predictionMessage");
    const noticeEl      = document.querySelector(".demo-notice");

    /* Show loading */
    statusEl.textContent      = "ANALYZING...";
    statusEl.style.color      = "#98a2b3";
    probEl.textContent        = "—";
    msgEl.textContent         = "Calling the CreditFair ML model...";

    try {

        const response = await fetch("/predict", {
            method:  "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                loan_amnt:      loanAmount,
                int_rate:       interestRate,
                annual_inc:     annualIncome,
                home_ownership: homeOwnership,
                purpose:        purpose,
                dti:            dti
            })
        });

        if (!response.ok) throw new Error("API error " + response.status);

        const result = await response.json();

        const fp = result.probability_fully_paid;
        const co = result.probability_charged_off;

        if (result.predicted_class === 1) {
            statusEl.textContent = "CHARGED OFF";
            statusEl.style.color = "#fb7185";
            probEl.textContent   = co.toFixed(2) + "%";
            msgEl.textContent    = "The CreditFair ML model predicts a higher default risk for these inputs.";
        } else {
            statusEl.textContent = "FULLY PAID";
            statusEl.style.color = "#a5b4fc";
            probEl.textContent   = fp.toFixed(2) + "%";
            msgEl.textContent    = "The CreditFair ML model predicts this loan is likely to be fully repaid.";
        }

        if (noticeEl) {
            noticeEl.innerHTML =
                "<strong>Live ML Mode:</strong> Result produced by the deployed " +
                "scikit-learn Logistic Regression model via the FastAPI backend.";
        }

        createPredictionChart(fp, co);

    } catch (err) {

        /* Fallback demo scoring when API is unreachable */
        console.warn("API unavailable – demo mode:", err);

        if (noticeEl) {
            noticeEl.innerHTML =
                "<strong>Presentation Mode:</strong> Backend API not reachable. " +
                "Showing demo score instead.";
        }

        let riskScore = 0;

        if (interestRate >= 15)       riskScore += 18;
        else if (interestRate >= 12)  riskScore += 10;
        else if (interestRate >= 8)   riskScore += 4;

        if (dti >= 30)       riskScore += 20;
        else if (dti >= 20)  riskScore += 10;
        else                 riskScore += 2;

        if (annualIncome > 0) {
            const ratio = loanAmount / annualIncome;
            if (ratio > 0.5)       riskScore += 18;
            else if (ratio > 0.3)  riskScore += 10;
            else                   riskScore += 3;
        }

        if (homeOwnership === "RENT")     riskScore += 5;
        else if (homeOwnership === "OTHER")   riskScore += 8;
        else if (homeOwnership === "OWN")     riskScore += 3;
        else if (homeOwnership === "MORTGAGE") riskScore += 2;

        if (purpose === "debt_consolidation") riskScore += 3;
        if (purpose === "major_purchase")     riskScore += 2;

        let co = Math.max(5, Math.min(95, 20 + riskScore * 0.65));
        let fp = 100 - co;

        if (co >= 50) {
            statusEl.textContent = "CHARGED OFF";
            statusEl.style.color = "#fb7185";
            probEl.textContent   = co.toFixed(2) + "%";
            msgEl.textContent    = "Demo: higher relative default risk for these inputs.";
        } else {
            statusEl.textContent = "FULLY PAID";
            statusEl.style.color = "#a5b4fc";
            probEl.textContent   = fp.toFixed(2) + "%";
            msgEl.textContent    = "Demo: higher relative probability of full repayment.";
        }

        createPredictionChart(fp, co);
    }

}


/* =========================================================
   SCROLL ANIMATION
========================================================= */

const observer =
    new IntersectionObserver(
        function(entries) {

            entries.forEach(
                function(entry) {

                    if (entry.isIntersecting) {

                        entry.target.classList.add(
                            "visible"
                        );

                    }

                }
            );

        },
        {
            threshold: 0.12
        }
    );


document
    .querySelectorAll(
        ".stat-card, .chart-card, .metric-card, .workflow-item"
    )
    .forEach(
        function(element) {

            observer.observe(element);

        }
    );