from openai import OpenAI
import yaml
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_openai_api():
    try:
        # Load config
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)

        # Initialize OpenAI client
        client = OpenAI(api_key=config["api_keys"]["openai"])

        # Make a simple API call
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello, this is a test message."},
            ],
            max_tokens=50,
        )

        print("API Test Successful!")
        print("Response:", response.choices[0].message.content)
        return True

    except Exception as e:
        print(f"Error testing API: {str(e)}")
        return False


if __name__ == "__main__":
    test_openai_api()
