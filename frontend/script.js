// ============================================
// Tab Navigation
// ============================================
const tabButtons = document.querySelectorAll('.tab-btn');
const tabContents = document.querySelectorAll('.tab-content');

tabButtons.forEach(btn => {
    btn.addEventListener('click', () => {
        // Remove active class from all buttons and contents
        tabButtons.forEach(b => b.classList.remove('active'));
        tabContents.forEach(c => c.classList.remove('active'));
        
        // Add active class to clicked button and corresponding content
        btn.classList.add('active');
        const tabId = btn.dataset.tab;
        document.getElementById(`${tabId}-tab`).classList.add('active');
        
        // Load data for the tab
        if (tabId === 'feed') {
            loadFeed();
        } else if (tabId === 'dashboard') {
            loadDashboard();
        }
    });
});

// ============================================
// Analyze Text Tab
// ============================================
const messageInput = document.getElementById("messageInput");
const predictBtn = document.getElementById("predictBtn");
const result = document.getElementById("result");
const alertBox = document.getElementById("alertBox");
const errorEl = document.getElementById("error");
const predictionLabel = document.getElementById("predictionLabel");
const confidenceScore = document.getElementById("confidenceScore");
const resultBadge = document.getElementById("resultBadge");

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

        // Display prediction
        predictionLabel.textContent = data.prediction;
        confidenceScore.textContent = data.confidence.toFixed(2);
        result.classList.remove("hidden");
        
        // Set badge color based on prediction
        const label = data.prediction.toLowerCase();
        resultBadge.textContent = label;
        resultBadge.className = 'badge ' + label;
        
        // Show alert if toxicity is high (bullying or offensive)
        if (label === 'bullying' || label === 'offensive') {
            alertBox.classList.remove("hidden");
        } else {
            alertBox.classList.add("hidden");
        }
        
    } catch (error) {
        showError("Unable to connect to the server. Please try again.");
        console.error(error);
    }
});

function showError(message) {
    errorEl.textContent = message;
    errorEl.classList.remove("hidden");
    result.classList.add("hidden");
    alertBox.classList.add("hidden");
}

function clearMessages() {
    errorEl.textContent = "";
    errorEl.classList.add("hidden");
}

// ============================================
// Live Feed Tab
// ============================================
const feedPosts = document.getElementById("feedPosts");
const feedLoading = document.getElementById("feedLoading");
const feedAlert = document.getElementById("feedAlert");
const refreshFeedBtn = document.getElementById("refreshFeedBtn");

// Feed stat elements
const feedTotal = document.getElementById("feedTotal");
const feedToxic = document.getElementById("feedToxic");
const feedRate = document.getElementById("feedRate");

async function loadFeed() {
    feedLoading.classList.remove("hidden");
    feedPosts.innerHTML = "";
    
    try {
        const response = await fetch("/feed?count=15");
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.error || "Failed to load feed");
        }
        
        // Update stats
        feedTotal.textContent = data.stats.total_comments;
        feedToxic.textContent = data.stats.toxic_count;
        feedRate.textContent = data.stats.toxicity_rate + "%";
        
        // Show alert if toxicity > 70%
        if (data.alert) {
            feedAlert.classList.remove("hidden");
        } else {
            feedAlert.classList.add("hidden");
        }
        
        // Render feed posts
        data.feed.forEach(post => {
            const postEl = createFeedPost(post);
            feedPosts.appendChild(postEl);
        });
        
    } catch (error) {
        feedPosts.innerHTML = `<div class="error-message">Error loading feed: ${error.message}</div>`;
    } finally {
        feedLoading.classList.add("hidden");
    }
}

function createFeedPost(post) {
    const div = document.createElement("div");
    const isToxic = post.label === "bullying" || post.label === "offensive";
    
    div.className = `feed-post ${isToxic ? 'toxic' : 'safe'}`;
    div.innerHTML = `
        <div class="post-header">
            <span class="username">${post.username}</span>
            <span class="timestamp">${post.timestamp}</span>
        </div>
        <div class="post-content">${post.text}</div>
        <div class="post-footer">
            <span class="label ${post.label}">${post.label}</span>
            <span class="metrics">❤️ ${post.likes} · 🔄 ${post.shares}</span>
        </div>
    `;
    
    return div;
}

refreshFeedBtn.addEventListener("click", loadFeed);

// ============================================
// Dashboard Tab
// ============================================
const refreshDashBtn = document.getElementById("refreshDashBtn");
const dashAlert = document.getElementById("dashAlert");

// Dashboard stat elements
const dashTotal = document.getElementById("dashTotal");
const dashToxic = document.getElementById("dashToxic");
const dashSafe = document.getElementById("dashSafe");
const dashPercentage = document.getElementById("dashPercentage");
const dashBullying = document.getElementById("dashBullying");
const dashOffensive = document.getElementById("dashOffensive");
const dashNeutral = document.getElementById("dashNeutral");

async function loadDashboard() {
    try {
        const response = await fetch("/dashboard");
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.error || "Failed to load dashboard");
        }
        
        // Update stats
        dashTotal.textContent = data.total_comments;
        dashToxic.textContent = data.toxic_comments;
        dashSafe.textContent = data.neutral_count;
        dashPercentage.textContent = data.toxicity_percentage + "%";
        
        // Update breakdown
        dashBullying.textContent = data.bullying_count;
        dashOffensive.textContent = data.offensive_count;
        dashNeutral.textContent = data.neutral_count;
        
        // Show alert if toxicity > 70%
        if (data.alert) {
            dashAlert.classList.remove("hidden");
            document.getElementById("dashAlertMessage").textContent = data.alert_message;
        } else {
            dashAlert.classList.add("hidden");
        }
        
    } catch (error) {
        console.error("Error loading dashboard:", error);
    }
}

refreshDashBtn.addEventListener("click", loadDashboard);

// ============================================
// Initial Load
// ============================================
// Load dashboard by default on page load
loadDashboard();
