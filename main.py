from pymilvus import MilvusClient
import hashlib

def text_to_vector(text, dim=8):
    # Simple hash-based text to vector conversion
    hash_object = hashlib.md5(text.encode())
    hash_digest = hash_object.digest()
    return [float(byte) / 255 for byte in hash_digest[:dim]]

# Initialize Milvus client
client = MilvusClient("hello_world.db")

# Create a collection
collection_name = "hello_world_collection"
vector_dim = 8  # We'll use 8-dimensional vectors for this example
if client.has_collection(collection_name):
    client.drop_collection(collection_name)
client.create_collection(
    collection_name=collection_name,
    dimension=vector_dim
)

# Prepare some sample data
texts = ["Hello, how are you?", "World of AI is fascinating", "Milvus is a vector database"]
data = [
    {"id": i, "vector": text_to_vector(text, vector_dim), "text": text}
    for i, text in enumerate(texts)
]

# Insert the data
client.insert(collection_name=collection_name, data=data)

# Perform a search
query_text = "Tell me about databases"
query_vector = text_to_vector(query_text, vector_dim)
search_results = client.search(
    collection_name=collection_name,
    data=[query_vector],
    limit=3,
    output_fields=["text"]
)

# Print the results
print(f"Query: {query_text}")
print("Search Results:")
for result in search_results[0]:
    print(f"ID: {result['id']}, Text: {result['entity']['text']}, Distance: {result['distance']}")

# Clean up
client.drop_collection(collection_name)