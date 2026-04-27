"""
Social Media Data Collector
Collects posts and comments from various social media platforms.
"""

import os
import time
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from datetime import datetime
import pandas as pd
import requests

logger = logging.getLogger(__name__)


class SocialMediaCollector(ABC):
    """Base class for social media data collection."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.collected_data = []

    @abstractmethod
    def authenticate(self) -> bool:
        """Authenticate with the social media platform."""
        pass

    @abstractmethod
    def fetch_posts(self, query: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Fetch posts matching the query."""
        pass

    @abstractmethod
    def fetch_comments(self, post_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Fetch comments for a specific post."""
        pass

    def save_to_csv(self, filepath: str, platform: str):
        """Save collected data to CSV file."""
        if self.collected_data:
            df = pd.DataFrame(self.collected_data)
            df["platform"] = platform
            df["collected_at"] = datetime.utcnow().isoformat()
            
            # Append or create new file
            mode = "a" if os.path.exists(filepath) else "w"
            header = mode == "w"
            df.to_csv(filepath, mode=mode, header=header, index=False)
            logger.info(f"Saved {len(self.collected_data)} records to {filepath}")


class TwitterCollector(SocialMediaCollector):
    """Collector for Twitter/X posts."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_key = config.get("twitter_api_key")
        self.api_secret = config.get("twitter_api_secret")
        self.access_token = config.get("twitter_access_token")
        self.access_secret = config.get("twitter_access_secret")
        self.bearer_token = config.get("twitter_bearer_token")
        self.client = None

    def authenticate(self) -> bool:
        """Authenticate using Twitter API v2."""
        try:
            import tweepy
            self.client = tweepy.Client(
                bearer_token=self.bearer_token,
                consumer_key=self.api_key,
                consumer_secret=self.api_secret,
                access_token=self.access_token,
                access_token_secret=self.access_secret
            )
            # Test authentication
            self.client.get_me()
            logger.info("Twitter authentication successful")
            return True
        except Exception as e:
            logger.error(f"Twitter authentication failed: {e}")
            return False

    def fetch_posts(self, query: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Fetch tweets matching the query."""
        if not self.client:
            raise RuntimeError("Not authenticated. Call authenticate() first.")

        tweets = []
        try:
            response = self.client.search_recent_tweets(
                query=query,
                max_results=min(limit, 100),
                tweet_fields=["created_at", "public_metrics", "text"]
            )
            
            if response.data:
                for tweet in response.data:
                    tweets.append({
                        "id": tweet.id,
                        "text": tweet.text,
                        "created_at": tweet.created_at.isoformat() if tweet.created_at else None,
                        "likes": tweet.public_metrics.get("like_count", 0) if tweet.public_metrics else 0,
                        "retweets": tweet.public_metrics.get("retweet_count", 0) if tweet.public_metrics else 0,
                    })
            logger.info(f"Fetched {len(tweets)} tweets for query: {query}")
        except Exception as e:
            logger.error(f"Error fetching tweets: {e}")

        self.collected_data.extend(tweets)
        return tweets

    def fetch_comments(self, post_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Fetch replies to a specific tweet."""
        if not self.client:
            raise RuntimeError("Not authenticated. Call authenticate() first.")

        comments = []
        try:
            response = self.client.get_tweet(
                post_id,
                expansions=["author_id"],
                tweet_fields=["created_at", "text"]
            )
            
            # Note: This is a simplified version. Full implementation
            # would need pagination for full comment threads.
            if response.data:
                comments.append({
                    "id": response.data.id,
                    "text": response.data.text,
                    "created_at": response.data.created_at.isoformat() if response.data.created_at else None,
                })
        except Exception as e:
            logger.error(f"Error fetching comments: {e}")

        self.collected_data.extend(comments)
        return comments


class RedditCollector(SocialMediaCollector):
    """Collector for Reddit posts and comments."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.client_id = config.get("reddit_client_id")
        self.client_secret = config.get("reddit_client_secret")
        self.user_agent = config.get("reddit_user_agent", "CyberbullyingDetector/1.0")
        self.username = config.get("reddit_username")
        self.password = config.get("reddit_password")
        self.reddit = None

    def authenticate(self) -> bool:
        """Authenticate with Reddit."""
        try:
            import praw
            self.reddit = praw.Reddit(
                client_id=self.client_id,
                client_secret=self.client_secret,
                user_agent=self.user_agent,
                username=self.username,
                password=self.password
            )
            # Verify authentication
            _ = self.reddit.user.me()
            logger.info("Reddit authentication successful")
            return True
        except Exception as e:
            logger.error(f"Reddit authentication failed: {e}")
            return False

    def fetch_posts(self, query: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Fetch Reddit posts matching the query."""
        if not self.reddit:
            raise RuntimeError("Not authenticated. Call authenticate() first.")

        posts = []
        try:
            subreddit = self.reddit.subreddit("all")
            for post in subreddit.search(query, limit=limit):
                posts.append({
                    "id": post.id,
                    "title": post.title,
                    "text": post.selftext,
                    "created_utc": post.created_utc,
                    "score": post.score,
                    "num_comments": post.num_comments,
                    "subreddit": str(post.subreddit)
                })
            logger.info(f"Fetched {len(posts)} Reddit posts for query: {query}")
        except Exception as e:
            logger.error(f"Error fetching Reddit posts: {e}")

        self.collected_data.extend(posts)
        return posts

    def fetch_comments(self, post_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Fetch comments for a specific Reddit post."""
        if not self.reddit:
            raise RuntimeError("Not authenticated. Call authenticate() first.")

        comments = []
        try:
            submission = self.reddit.submission(id=post_id)
            submission.comments.replace_more(limit=None)
            
            for comment in submission.comments[:limit]:
                comments.append({
                    "id": comment.id,
                    "text": comment.body,
                    "created_utc": comment.created_utc,
                    "score": comment.score,
                    "parent_id": comment.parent_id
                })
            logger.info(f"Fetched {len(comments)} comments for post: {post_id}")
        except Exception as e:
            logger.error(f"Error fetching Reddit comments: {e}")

        self.collected_data.extend(comments)
        return comments


class GenericAPICollector(SocialMediaCollector):
    """Collector for generic REST APIs (Facebook, Instagram, etc.)."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.base_url = config.get("api_base_url")
        self.api_token = config.get("api_token")
        self.headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json"
        }

    def authenticate(self) -> bool:
        """Test API connection."""
        try:
            if self.base_url:
                response = requests.get(
                    f"{self.base_url}/me",
                    headers=self.headers,
                    timeout=10
                )
                return response.status_code == 200
            return False
        except Exception as e:
            logger.error(f"API authentication failed: {e}")
            return False

    def fetch_posts(self, query: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Fetch posts from generic API."""
        posts = []
        try:
            # Generic implementation - customize per platform
            response = requests.get(
                f"{self.base_url}/posts",
                headers=self.headers,
                params={"q": query, "limit": limit},
                timeout=30
            )
            if response.status_code == 200:
                data = response.json()
                posts = data.get("data", [])
        except Exception as e:
            logger.error(f"Error fetching posts: {e}")

        self.collected_data.extend(posts)
        return posts

    def fetch_comments(self, post_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Fetch comments from generic API."""
        comments = []
        try:
            response = requests.get(
                f"{self.base_url}/posts/{post_id}/comments",
                headers=self.headers,
                params={"limit": limit},
                timeout=30
            )
            if response.status_code == 200:
                data = response.json()
                comments = data.get("data", [])
        except Exception as e:
            logger.error(f"Error fetching comments: {e}")

        self.collected_data.extend(comments)
        return comments


def create_collector(platform: str, config: Dict[str, Any]) -> SocialMediaCollector:
    """Factory function to create the appropriate collector."""
    collectors = {
        "twitter": TwitterCollector,
        "x": TwitterCollector,
        "reddit": RedditCollector,
        "generic": GenericAPICollector,
    }
    
    collector_class = collectors.get(platform.lower())
    if not collector_class:
        raise ValueError(f"Unsupported platform: {platform}")
    
    return collector_class(config)