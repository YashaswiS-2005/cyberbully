"""
Feed Simulation Module
Simulates real-time social media feed with sample comments from dataset.
"""

import os
import random
import time
from datetime import datetime
from typing import List, Dict, Any
import pandas as pd

# Path to dataset
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "dataset", "cyberbullying_dataset.csv")

# Sample usernames for simulation
USERNAMES = [
    "@user_123", "@cyber_surfer", "@netizen_99", "@digital_native", 
    "@online_hero", "@web_warrior", "@social_king", "@tech_fan",
    "@pixel_master", "@byte_runner", "@data_diver", "@cloud_ninja",
    "@stream_star", "@feed_follower", "@post_pioneer", "@trend_setter"
]

# Sample timestamps (simulating recent posts)
def get_random_timestamp() -> str:
    """Generate a random timestamp within the last hour."""
    import datetime as dt
    now = dt.datetime.now()
    random_seconds = random.randint(0, 3600)
    return (now - dt.timedelta(seconds=random_seconds)).strftime("%Y-%m-%d %H:%M:%S")


class FeedSimulator:
    """Simulates a real-time social media feed."""
    
    def __init__(self, dataset_path: str = None):
        self.dataset_path = dataset_path or DATA_PATH
        self.comments = []
        self.load_dataset()
    
    def load_dataset(self):
        """Load comments from the dataset file."""
        try:
            if os.path.exists(self.dataset_path):
                df = pd.read_csv(self.dataset_path)
                if "text" in df.columns and "label" in df.columns:
                    self.comments = df.to_dict("records")
                    return
        except Exception as e:
            print(f"Error loading dataset: {e}")
        
        # Fallback sample comments if dataset not found
        self.comments = [
            {"text": "You're so stupid!", "label": "bullying"},
            {"text": "Have a great day!", "label": "neutral"},
            {"text": "This is ridiculous", "label": "offensive"},
            {"text": "Nobody likes you", "label": "bullying"},
            {"text": "Thanks for sharing", "label": "neutral"},
            {"text": "You should disappear", "label": "bullying"},
            {"text": "I hate this", "label": "offensive"},
            {"text": "Great work!", "label": "neutral"},
        ]
    
    def get_random_comment(self) -> Dict[str, Any]:
        """Get a random comment from the dataset."""
        comment = random.choice(self.comments)
        return {
            "id": random.randint(10000, 99999),
            "username": random.choice(USERNAMES),
            "text": comment["text"],
            "label": comment["label"],
            "timestamp": get_random_timestamp(),
            "likes": random.randint(0, 500),
            "shares": random.randint(0, 100)
        }
    
    def generate_feed(self, count: int = 10) -> List[Dict[str, Any]]:
        """Generate a feed with specified number of comments."""
        feed = []
        for _ in range(count):
            feed.append(self.get_random_comment())
        
        # Sort by timestamp (newest first)
        feed.sort(key=lambda x: x["timestamp"], reverse=True)
        return feed
    
    def get_toxic_comments(self, feed: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter toxic comments (bullying + offensive) from feed."""
        return [c for c in feed if c["label"] in ["bullying", "offensive"]]
    
    def calculate_toxicity_rate(self, feed: List[Dict[str, Any]]) -> float:
        """Calculate the percentage of toxic comments in feed."""
        if not feed:
            return 0.0
        toxic = len(self.get_toxic_comments(feed))
        return round((toxic / len(feed)) * 100, 2)


# Global simulator instance
_feed_simulator = None

def get_feed_simulator() -> FeedSimulator:
    """Get or create the global feed simulator instance."""
    global _feed_simulator
    if _feed_simulator is None:
        _feed_simulator = FeedSimulator()
    return _feed_simulator


def generate_live_feed(count: int = 10) -> Dict[str, Any]:
    """Generate a live feed and return with statistics."""
    simulator = get_feed_simulator()
    feed = simulator.generate_feed(count)
    
    toxic_comments = simulator.get_toxic_comments(feed)
    toxicity_rate = simulator.calculate_toxicity_rate(feed)
    
    return {
        "feed": feed,
        "stats": {
            "total_comments": len(feed),
            "toxic_count": len(toxic_comments),
            "toxicity_rate": toxicity_rate,
            "bullying_count": len([c for c in feed if c["label"] == "bullying"]),
            "offensive_count": len([c for c in feed if c["label"] == "offensive"]),
            "neutral_count": len([c for c in feed if c["label"] == "neutral"])
        },
        "alert": toxicity_rate > 70,
        "alert_message": "Warning: This content may be harmful" if toxicity_rate > 70 else None
    }


if __name__ == "__main__":
    # Test the feed simulator
    result = generate_live_feed(10)
    print(f"Generated {result['stats']['total_comments']} comments")
    print(f"Toxicity rate: {result['stats']['toxicity_rate']}%")
    print(f"Alert: {result['alert']}")