#Here we test that the Opean AI API key is working and we print the resukt of the question "What is the meaning of life?"
#Import the Open AI API and the os module
import openai
import os
import dotenv
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()


#Load Open AI API Key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
#Set the Open AI API key
openai.api_key = OPENAI_API_KEY
#Ask the question "What is the meaning of life?"
response = openai.Completion.create(
  engine="davinci",
  prompt="What is the meaning of life?",
  max_tokens=5
)
#Print the response
print(response)

