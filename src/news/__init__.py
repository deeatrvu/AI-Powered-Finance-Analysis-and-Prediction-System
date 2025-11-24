"""
News collection and processing module.
"""

from .collector import NewsCollector
from .summarizer import NewsSummarizer

__all__ = ["NewsCollector", "NewsSummarizer"]
