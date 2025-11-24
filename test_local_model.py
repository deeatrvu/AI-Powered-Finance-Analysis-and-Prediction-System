from src.news.summarizer import NewsSummarizer
import yaml
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_local_models():
    try:
        # Load config
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)

        # Initialize summarizer
        summarizer = NewsSummarizer(config)

        # Test text
        test_text = """
        Apple Inc. reported record-breaking quarterly earnings, with iPhone sales exceeding expectations. 
        The company's services division also showed strong growth, contributing to a 20% increase in overall revenue. 
        Analysts predict continued growth in the coming quarters, driven by new product launches and expanding market share.
        """

        # Test summarization
        print("\nTesting Summarization:")
        summary = summarizer.summarize(test_text)
        print("Summary:", summary)

        # Test impact analysis
        print("\nTesting Impact Analysis:")
        impact = summarizer.analyze_impact(test_text)
        print("Impact Analysis:", impact)

        return True

    except Exception as e:
        print(f"Error testing local models: {str(e)}")
        return False


if __name__ == "__main__":
    test_local_models()
