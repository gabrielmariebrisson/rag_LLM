# Imports
import os
from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import AzureSearch
from langchain.document_loaders import CSVLoader
from langchain.text_splitter import CharacterTextSplitter
import openai
from pprint import pprint

# Load environment variables
load_dotenv('.env')

# Initialize embeddings
embeddings = OpenAIEmbeddings(
    deployment="demo-embedding",
    chunk_size=1
)

# Connect to Azure Cognitive Search
acs = AzureSearch(
    azure_search_endpoint=os.getenv('SEARCH_SERVICE_NAME'),
    azure_search_key=os.getenv('SEARCH_API_KEY'),
    index_name=os.getenv('SEARCH_INDEX_NAME'),
    embedding_function=embeddings.embed_query
)

# Load CSV documents
loader = CSVLoader("wine-ratings.csv")
documents = loader.load()

# Split documents into chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

# Add documents to Azure Cognitive Search
acs.add_documents(documents=docs)

# Similarity search example
docs = acs.similarity_search_with_relevance_scores(
    query="What is the best Cabernet Sauvignon wine in Napa Valley above 94 points?",
    k=5
)

# Inspect top document
print(docs[0][0].page_content)
print(dir(docs[0][0]))

# Set up OpenAI Azure API
openai.api_base = os.getenv("OPENAI_API_BASE")      # Azure OpenAI endpoint
openai.api_key = os.getenv("OPENAI_API_KEY")        # Azure OpenAI key
openai.api_type = "azure"
openai.api_version = "2023-05-15"

# Prepare messages for ChatCompletion
messages = [
    {"role": "system", "content": "Assistant is a chatbot that helps you find the best wine for your taste."},
    {"role": "user", "content": "What is the best Cabernet Sauvignon wine in Napa Valley above 94 points?"},
    {"role": "assistant", "content": docs[0][0].page_content}
]

# Get response from Azure OpenAI
response = openai.ChatCompletion.create(
    engine="demo-alfredo",
    messages=messages
)

# Print response
pprint(response)
print(response['choices'][0]['message']['content'])

# Another similarity search
docs = acs.similarity_search_with_relevance_scores(
    query="What is the best Pinot Noir wine in Oregon above 94 points?",
    k=5
)

print(docs[0][0].page_content)
print(dir(docs[0][0]))
