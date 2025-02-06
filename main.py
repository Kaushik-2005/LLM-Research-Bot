import serpapi
import os

from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv('SERPAPI_KEY')
client = serpapi.Client(api_key=api_key)

query = input("enter a input:")

result = client.search(
    q=query,
    engine="google_scholar",
    hl="en",
    gl="us"
)

print(result)