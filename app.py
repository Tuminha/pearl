from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import TokenTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import os
import openai  # Import the OpenAI library


# Load environment variables from .env file
load_dotenv()

# Load Open AI API Key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY  # <-- Add this line


# Load the PDF document using PyPDFLoader
loader = PyPDFLoader("docs/fdmed_04_1085251.pdf")

# Extract the text data from the PDF document
data = loader.load()

# Split the text into chunks using TokenTextSplitter
text_splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=150)
chunks = text_splitter.split_documents(data)

# Generate embeddings using OpenAIEmbeddings
embeddings = OpenAIEmbeddings()

# Create a retriever using Chroma and the generated embeddings
retriever = Chroma.from_documents(chunks, embeddings).as_retriever()

# Initialize the ChatOpenAI model
llm = ChatOpenAI(model="gpt-3.5-turbo-16k", temperature=0.1)

# Create a RetrievalQA instance with the ChatOpenAI model and the retriever
qa = RetrievalQA.from_chain_type(llm, chain_type="stuff", retriever=retriever)

# Define the PEARL strategy function
# Define the PEARL strategy function
def pearl_strategy(question, context, llm):
    # PEARL Decomposition: Break down the question into sub-questions
    sub_questions = [
        f"Step 1: What is the main theme of the article regarding {question}?",
        f"Step 2: What are the key points in the article related to {question}?",
        f"Step 3: What actions does the article suggest about {question}?"
    ]
    
    # Execute the steps and collect the answers
    answers = []
    for sub_q in sub_questions:
        prompt = f"{sub_q}\n{context}"
        
        # Replace the following line with the correct method for generating text
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-16k",  # replace with the model you are using
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        answer = response.choices[0].message.content.strip()
        
        answers.append(answer)

    # Compile the answers into a coherent summary
    summary = " ".join(answers)
    return f"Question: {question}\nPEARL Summary: {summary}"


# Predefined questions for PEARL analysis
predefined_questions = [
    "What is the article about?",
    "What methodologies are used?",
    "What are the key findings?",
    "What recommendations are made?"
]

# Automated PEARL Analysis
for question in predefined_questions:
    # Use the retriever to get relevant chunks
    retrieved_docs = retriever.get_relevant_documents(question)
    
    # Concatenate the retrieved chunks to form the context
    context = " ".join([doc.page_content for doc in retrieved_docs])
    
    # Use the PEARL strategy to answer the question
    pearl_summary = pearl_strategy(question, context, llm)

    print(pearl_summary)
    print("------\n")

    #Export the output to a text file to the system downloads folder
    with open(os.path.join(os.path.expanduser("~"), "Downloads", "pearl_summary.txt"), "a") as f:
        f.write(pearl_summary)
        f.write("\n------\n")

