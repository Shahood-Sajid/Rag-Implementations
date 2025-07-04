import fitz
import os
import numpy as np
import json
from sentence_transformers import SentenceTransformer
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=GEMINI_API_KEY)

def extract_text_from_pdf(pdf_path):
    """
    Extracts text from a PDF file and prints the first `num_chars` characters.

    Args:
    pdf_path (str): Path to the PDF file.

    Returns:
    str: Extracted text from the PDF.
    """
    # Open the PDF file
    mypdf = fitz.open(pdf_path)
    all_text = ""  # Initialize an empty string to store the extracted text

    # Iterate through each page in the PDF
    for page_num in range(mypdf.page_count):
        page = mypdf[page_num]  # Get the page
        text = page.get_text("text")  # Extract text from the page
        all_text += text  # Append the extracted text to the all_text string

    return all_text  # Return the extracted text

def chunk_text(text, n, overlap):
    """
    Chunks the given text into segments of n characters with overlap.

    Args:
    text (str): The text to be chunked.
    n (int): The number of characters in each chunk.
    overlap (int): The number of overlapping characters between chunks.

    Returns:
    List[str]: A list of text chunks.
    """
    chunks = []  # Initialize an empty list to store the chunks
    
    # Loop through the text with a step size of (n - overlap)
    for i in range(0, len(text), n - overlap):
        # Append a chunk of text from index i to i + n to the chunks list
        chunks.append(text[i:i + n])

    return chunks  # Return the list of text chunks

def create_embeddings(text_chunks):
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    embed = model.encode(text_chunks)

    return embed

def cosine_similarity(vec1, vec2):
    """
    Calculates the cosine similarity between two vectors.

    Args:
    vec1 (np.ndarray): The first vector.
    vec2 (np.ndarray): The second vector.

    Returns:
    float: The cosine similarity between the two vectors.
    """
    # Compute the dot product of the two vectors and divide by the product of their norms
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def semantic_search(query, text_chunks, embeddings, k=5):
    """
    Performs semantic search on the text chunks using the given query and embeddings.

    Args:
    query (str): The query for the semantic search.
    text_chunks (List[str]): A list of text chunks to search through.
    embeddings (List[dict]): A list of embeddings for the text chunks.
    k (int): The number of top relevant text chunks to return. Default is 5.

    Returns:
    List[str]: A list of the top k most relevant text chunks based on the query.
    """
    # Create an embedding for the query
    query_embedding = create_embeddings(query)
    similarity_scores = []  # Initialize a list to store similarity scores

    # Calculate similarity scores between the query embedding and each text chunk embedding
    for i, chunk_embedding in enumerate(embeddings):
        similarity_score = cosine_similarity(np.array(query_embedding), np.array(chunk_embedding))
        similarity_scores.append((i, similarity_score))  # Append the index and similarity score

    # Sort the similarity scores in descending order
    similarity_scores.sort(key=lambda x: x[1], reverse=True)
    # Get the indices of the top k most similar text chunks
    top_indices = [index for index, _ in similarity_scores[:k]]
    # Return the top k most relevant text chunks
    return [text_chunks[index] for index in top_indices]
    
# Define the path to the PDF file
pdf_path = "data/AI_Information.pdf"

# Extract text from the PDF file
extracted_text = extract_text_from_pdf(pdf_path)

# Chunk the extracted text into segments of 1000 characters with an overlap of 200 characters
text_chunks = chunk_text(extracted_text, 1000, 200)

vector_embeded_text = create_embeddings(text_chunks)

# Load the validation data from a JSON file
with open('data/val.json') as f:
    data = json.load(f)

# Extract the first query from the validation data
query = data[3]['question']


# Perform semantic search to find the top 2 most relevant text chunks for the query
top_chunks = semantic_search(query, text_chunks, vector_embeded_text, k=2)

# Define the system prompt for the AI assistant
system_prompt = "You are an AI assistant that strictly answers based on the given context. If the answer cannot be derived directly from the provided context, respond with: 'I do not have enough information to answer that.'"

def generate_response(system_prompt, user_message):
    """
    Generates a response from the AI model based on the system prompt and user message.

    Args:
    system_prompt (str): The system prompt to guide the AI's behavior.
    user_message (str): The user's message or query.

    Returns:
    string: The response from the AI model.
    """
    response = client.models.generate_content(
    model='gemini-2.0-flash',
    contents = user_message,
    config=types.GenerateContentConfig(system_instruction = system_prompt))

    return response.text

# Create the user prompt based on the top chunks
user_prompt = "\n".join([f"Context {i + 1}:\n{chunk}\n=====================================\n" for i, chunk in enumerate(top_chunks)])
user_prompt = f"{user_prompt}\nQuestion: {query}"

print("user prompt: ", user_prompt)

# Generate AI response
ai_response = generate_response(system_prompt, user_prompt)

print("ai response: ", ai_response)