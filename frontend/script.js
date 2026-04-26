const messageInput = document.getElementById("messageInput");
const predictBtn = document.getElementById("predictBtn");
const result = document.getElementById("result");
const statsSection = document.getElementById("stats");
const errorEl = document.getElementById("error");
const predictionLabel = document.getElementById("predictionLabel");
const confidenceScore = document.getElementById("confidenceScore");
const bullyingCount = document.getElementById("bullyingCount");
const offensiveCount = document.getElementById("offensiveCount");
const neutralCount = document.getElementById("neutralCount");
const totalCount = document.getElementById("totalCount");

predictBtn.addEventListener("click", async () => {
    const text = messageInput.value.trim();
    if (!text) {
        showError("Please enter a message to analyze.");
        return;
    }

    try {
        clearMessages();
        const response = await fetch("/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({ text }),
        });

        const data = await response.json();
        if (!response.ok) {
            showError(data.error || "Prediction failed.");
            return;
        }

        predictionLabel.textContent = data.prediction;
        confidenceScore.textContent = data.confidence.toFixed(2);
        result.classList.remove("hidden");
        await refreshStats();
    } catch (error) {
        showError("Unable to connect to the server. Please try again.");
        console.error(error);
    }
});

async function refreshStats() {
    try {
        const response = await fetch("/stats");
        const data = await response.json();
        if (response.ok) {
            bullyingCount.textContent = data.counts.bullying;
            offensiveCount.textContent = data.counts.offensive;
            neutralCount.textContent = data.counts.neutral;
            totalCount.textContent = data.total;
            statsSection.classList.remove("hidden");
        }
    } catch (error) {
        console.warn("Stats endpoint unavailable.", error);
    }
}

function showError(message) {
    errorEl.textContent = message;
    errorEl.classList.remove("hidden");
    result.classList.add("hidden");
}

function clearMessages() {
    errorEl.textContent = "";
    errorEl.classList.add("hidden");
}

refreshStats();
