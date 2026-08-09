import chromadb
from chromadb.config import Settings
import time
import json
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from litellm import completion

EMBEDDING_CONFIGS = {
    "local_embeddings": {
        "provider": "sentence_transformers",
        "model": "all-MiniLM-L6-v2",
        "dimensions": 384,
        "description": "Local embeddings for cost-effective processing"
    }
}

# Create configurations for technical docs, FAQ support, and knowledge base
COLLECTION_CONFIGS = {
    # Technical documentation collection
    "tech_docs": {
        "name": "technical_documentation",
        "metadata_fields": ["source", "category", "difficulty", "last_updated"],
        "description": "Technical documentation with structured metadata"
    },
    # FAQ collection for customer support
    "faq_support": {
        "name": "faq_customer_support", 
        "metadata_fields": ["category", "priority", "department", "tags"],
        "description": "FAQ database for customer support automation"
    },
    # Knowledge base for general information
    "knowledge_base": {
        "name": "general_knowledge",
        "metadata_fields": ["topic", "source", "confidence", "date_added"],
        "description": "General knowledge base for information retrieval"
    }
}

# Define realistic business documents with content and metadata
SAMPLE_DOCUMENTS = {
    "tech_docs": [
        {
            "id": "tech_001",
            "content": "ChromaDB is an open-source vector database designed for AI applications. It provides efficient storage and retrieval of high-dimensional vectors, making it ideal for semantic search, recommendation systems, and RAG implementations. ChromaDB supports multiple embedding functions and offers both in-memory and persistent storage options.",
            "metadata": {
                "source": "ChromaDB Documentation",
                "category": "Database",
                "difficulty": "Intermediate",
                "last_updated": "2024-01-15"
            }
        },
        {
            "id": "tech_002", 
            "content": "Retrieval-Augmented Generation (RAG) combines the power of large language models with external knowledge retrieval. By retrieving relevant documents before generation, RAG systems can provide more accurate, up-to-date, and contextually relevant responses while reducing hallucinations and improving factual accuracy.",
            "metadata": {
                "source": "AI Research Papers",
                "category": "Machine Learning",
                "difficulty": "Advanced",
                "last_updated": "2024-02-01"
            }
        },
        {
            "id": "tech_003",
            "content": "Vector embeddings are numerical representations of text that capture semantic meaning. Modern embedding models like OpenAI's text-embedding-3-small can convert text into high-dimensional vectors where similar concepts are positioned closer together in the vector space, enabling semantic search capabilities.",
            "metadata": {
                "source": "Embedding Guide",
                "category": "NLP",
                "difficulty": "Intermediate", 
                "last_updated": "2024-01-20"
            }
        }
    ],
    "faq_support": [
        {
            "id": "faq_001",
            "content": "Q: How do I reset my password? A: To reset your password, click on the 'Forgot Password' link on the login page, enter your email address, and follow the instructions sent to your email. The reset link expires after 24 hours for security purposes.",
            "metadata": {
                "category": "Account Management",
                "priority": "High",
                "department": "IT Support",
                "tags": "password, security, login"
            }
        },
        {
            "id": "faq_002",
            "content": "Q: What are your business hours? A: Our customer support is available Monday through Friday, 9 AM to 6 PM EST. For urgent technical issues, our emergency support line is available 24/7 for premium customers.",
            "metadata": {
                "category": "General Information",
                "priority": "Medium",
                "department": "Customer Service",
                "tags": "hours, support, availability"
            }
        },
        {
            "id": "faq_003",
            "content": "Q: How do I upgrade my subscription? A: You can upgrade your subscription by logging into your account, navigating to the 'Billing' section, and selecting 'Upgrade Plan'. Changes take effect immediately, and you'll be prorated for the current billing period.",
            "metadata": {
                "category": "Billing",
                "priority": "High", 
                "department": "Sales",
                "tags": "subscription, billing, upgrade"
            }
        }
    ]
}

class ChromaDBRAGSystem:
    """
    A comprehensive RAG system implementation using ChromaDB for vector storage and retrieval.
    """
    
    def __init__(self, embedding_config: str = "local_embeddings", persist_directory: str = "./chroma_db"):
        """
        Initialize the ChromaDB RAG system with specified configuration.
        Args:
            embedding_config (str): Configuration key for embedding strategy
            persist_directory (str): Directory for persistent storage
        """
        self.embedding_config = EMBEDDING_CONFIGS[embedding_config]
        self.persist_directory = persist_directory
        
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,  # Disable telemetry for privacy
                allow_reset=True             # Allow database reset for development
            )
        )
        self.collections = {}
        
        print(f"ChromaDB RAG System initialized")

    def create_collection(self, collection_key: str):
        """
        Create a new ChromaDB collection with specified configuration.
        Args:
            collection_key (str): Key from COLLECTION_CONFIGS
            
        Returns:
            chromadb.Collection: The created collection object
        """
        if collection_key not in COLLECTION_CONFIGS:
            raise ValueError(f"Unknown collection configuration: {collection_key}")
            
        config = COLLECTION_CONFIGS[collection_key]
        collection_name = config["name"]
        
        print(f"\nCreating collection: {collection_name}")
        
        try:
            self.client.delete_collection(collection_name)
            print("Deleted existing collection")
        except Exception as e:
            print(f"Error creating collection: {str(e)}")
            
        collection = self.client.create_collection(name=collection_name,embedding_function=None,metadata={"description": config["description"]})
        self.collections[collection_key] = collection
        print(f" Collection created successfully")
        return collection
    
    def generate_embeddings(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for a list of texts using the configured embedding model.
        Args:
            texts (List[str]): List of texts to embed
            
        Returns:
            List[List[float]]: List of embedding vectors
        """
        print(f"Generating embeddings for {len(texts)} texts...")
        
        try:
            model = SentenceTransformer(self.embedding_config["model"])
            embeddings = model.encode(texts).tolist()
            return embeddings
        except Exception as e:
            print(f"Error generating embeddings: {str(e)}")
            raise

    def add_documents(self, collection_key: str, documents: list[dict]) -> None:
        """
        Add documents to a ChromaDB collection with embeddings and metadata.

        Args:
            collection_key (str): Key identifying the target collection
            documents (List[Dict]): List of document dictionaries with content and metadata
        """
        if collection_key not in self.collections:
            raise ValueError(f"Collection {collection_key} not found. Create it first.")
            
        collection = self.collections[collection_key]
        
        print(f"\n📄 Adding {len(documents)} documents to {collection.name}")
        
        texts = [doc["content"] for doc in documents]
        ids = [doc["id"] for doc in documents]
        metadatas = [doc["metadata"] for doc in documents]
        embeddings = self.generate_embeddings(texts)
        try:
            collection.add(
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )
            
            print(f"Successfully added {len(documents)} documents")
            print(f"Collection now contains: {collection.count()} documents")      
        except Exception as e:
            print(f"Error adding documents: {str(e)}")
            raise
    def search_documents(self,collection_key: str, query: str, n_results: int = 3,  metadata_filter: dict | None = None
                        ) -> dict:
        """
        Search for relevant documents using semantic similarity.
        Args:
            collection_key (str): Key identifying the collection to search
            query (str): Search query text
            n_results (int): Number of results to return
            metadata_filter (Optional[Dict]): Metadata filters to apply
        Returns:
            Dict: Search results with documents, distances, and metadata
        """
        if collection_key not in self.collections:
            raise ValueError(f"Collection {collection_key} not found")
            
        collection = self.collections[collection_key]
        
        print(f"\nSearching collection: {collection.name}")
        print(f"   Query: '{query}'")
        print(f"   Requesting: {n_results} results")
        if metadata_filter:
            print(f" Filters: {metadata_filter}")
        
        try:
            query_embeddings = self.generate_embeddings([query])
            
            if not query_embeddings:
                print("Failed to generate query embedding")
                return {"documents": [], "distances": [], "metadatas": []}
            
            results = collection.query(
                query_embeddings=query_embeddings,
                n_results=n_results,
                where=metadata_filter,
                include=["documents", "distances", "metadatas"]
            )
            
            print(f"✅ Found {len(results['documents'][0])} relevant documents")
            
            formatted_results = {
                "query": query,
                "n_results": len(results['documents'][0]),
                "results": []
            }
            
            for i in range(len(results['documents'][0])):
                formatted_results["results"].append({
                    "document": results['documents'][0][i],
                    "similarity_score": 1 - results['distances'][0][i], 
                    "metadata": results['metadatas'][0][i],
                    "id": results['ids'][0][i]
                })
            return formatted_results
        except Exception as e:
            print(f"Error searching documents: {str(e)}")
            return {"documents": [], "distances": [], "metadatas": []}


    def generate_rag_response(self, collection_key: str, query: str, n_context: int = 3,
                            model: str = "gemma3:4b") -> dict:
        """
        Generate a response using Retrieval-Augmented Generation.
        Args:
            collection_key (str): Collection to search for context
            query (str): User query to answer
            n_context (int): Number of context documents to retrieve
            model (str): OpenAI model to use for generation
            
        Returns:
            Dict: RAG response with context, answer, and metadata
        """
        print(f"\nGenerating RAG response")
        print(f"   Query: '{query}'")
        
        start_time = time.time()
        
        # Step 1: Retrieve relevant context
        search_results = self.search_documents(collection_key, query, n_context)
        
        if search_results is None or "results" not in search_results or not search_results["results"]:
            return {
                "query": query,
                "answer": "I couldn't find relevant information to answer your question.",
                "context": [],
                "generation_time": 0,
                "context_used": 0
            }
        
        context_documents = []
        for result in search_results["results"]:
            context_documents.append({
                "content": result["document"],
                "similarity": result["similarity_score"],
                "source": result["metadata"].get("source",result["id"])
            })
        
        context_text = "\n\n".join([
            f"Document {i+1} (Similarity: {doc['similarity']:.3f}):\n{doc['content']}"
            for i, doc in enumerate(context_documents)
        ])
        
        prompt = f"""Based on the following context documents, please answer the user's question. If the context doesn't contain enough information to answer the question completely, please say so and provide what information you can.
        Context Documents:{context_text}
        User Question: {query}
        Please provide a comprehensive answer based on the context provided:"""
        try:
            response = completion(
                model="ollama/gemma3:4b",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                api_base="http://localhost:11434",
                max_completion_tokens=500,
                temperature=0.7,
            )
            generation_time = time.time() - start_time
            
            rag_response = {
                "query": query,
                "answer": response.choices[0].message.content,
                "context": context_documents,
                "generation_time": round(generation_time, 2),
                "context_used": len(context_documents),
                "model_used": model,
                "tokens_used": response.usage.total_tokens
            }
            
            print(f"   RAG response generated successfully")
            print(f"   Generation time: {generation_time:.2f}s")
            print(f"   Context documents used: {len(context_documents)}")
            print(f"   Tokens used: {response.usage.total_tokens}")
            
            return rag_response
            
        except Exception as e:
            print(f" Error generating RAG response: {str(e)}")
            return {
                "query": query,
                "answer": f"Error generating response: {str(e)}",
                "context": context_documents,
                "generation_time": 0,
                "context_used": len(context_documents)
            }

    def display_rag_response(self, rag_response: dict) -> None:
        """
        Display RAG response in a formatted, readable way.

        Args:
            rag_response (Dict): RAG response dictionary from generate_rag_response
        """
        print(f"\n" + "="*80)
        print(f"RAG RESPONSE")
        print(f"="*80)
        
        print(f"\n QUESTION:")
        print(f"{rag_response['query']}")
        
        print(f"\n ANSWER:")
        print(f"   {rag_response['answer']}")
        
        print(f"\n CONTEXT SOURCES ({rag_response['context_used']} documents):")
        for i, context in enumerate(rag_response['context']):
            print(f"   {i+1}. Similarity: {context['similarity']:.3f} | Source: {context['source']}")
            print(f"      Preview: {context['content'][:100]}...")
        
        print(f"\n PERFORMANCE METRICS:")
        print(f"   Generation Time: {rag_response['generation_time']}s")
        print(f"   Model Used: {rag_response.get('model_used', 'Unknown')}")
        print(f"   Tokens Used: {rag_response.get('tokens_used', 'Unknown')}")
        print(f"   Context Documents: {rag_response['context_used']}")

def chrome_db_use():
    """    
    This function showcases the complete workflow from database setup through
    document ingestion to intelligent query answering using RAG.
    """
    print("ChromaDB RAG System ")
    print("="*60)
    
    rag_system = ChromaDBRAGSystem(
        embedding_config="local_embeddings",
        persist_directory="./test_chroma_db"
    )
    
    print("\n Setting up document collections...")
    rag_system.create_collection("tech_docs")
    rag_system.create_collection("faq_support")
    
    print("\n📄 Adding sample documents...")
    rag_system.add_documents("tech_docs", SAMPLE_DOCUMENTS["tech_docs"])
    rag_system.add_documents("faq_support", SAMPLE_DOCUMENTS["faq_support"])
    
    test_queries = [
        {
            "collection": "tech_docs",
            "query": "What is ChromaDB and how does it work?",
            "description": "Technical documentation query"
        },
        {
            "collection": "tech_docs", 
            "query": "How do vector embeddings enable semantic search?",
            "description": "Conceptual understanding query"
        },
        {
            "collection": "faq_support",
            "query": "I forgot my password, how can I reset it?",
            "description": "Customer support query"
        },
        {
            "collection": "faq_support",
            "query": "What are your business hours?",
            "description": "General information query"
        }
    ]
    
    print("\n Testing RAG system with various queries...")
    for i, test in enumerate(test_queries, 1):
        print(f"\n{'='*20} TEST QUERY {i}: {test['description']} {'='*20}")
        
        rag_response = rag_system.generate_rag_response(
            collection_key=test["collection"],
            query=test["query"],
            n_context=2
        )
        
        rag_system.display_rag_response(rag_response)
    
    print(f"\n ChromaDB RAG completed successfully!")
    print(f"   Collections created: {len(rag_system.collections)}")
    print(f"   Documents processed: {sum(len(docs) for docs in SAMPLE_DOCUMENTS.values())}")
    print(f"   Queries tested: {len(test_queries)}")

if __name__ == "__main__":
    chrome_db_use()
        
    rag_system = ChromaDBRAGSystem()
    rag_system.create_collection("tech_docs")
    rag_system.add_documents("tech_docs", SAMPLE_DOCUMENTS["tech_docs"])
    
    filtered_results = rag_system.search_documents(
        "tech_docs", 
        "database information",
        metadata_filter={"category": "Database"}
    )
    
    queries = ["What is RAG?", "How do embeddings work?", "ChromaDB features"]
    for query in queries:
        response = rag_system.generate_rag_response("tech_docs", query)
        rag_system.display_rag_response(response)
# TO DO:
# □ Add support for local embedding models using sentence-transformers
# □ Implement batch processing for large document collections
# □ Add metadata-based filtering and advanced search capabilities
# □ Create a web interface for the RAG system using Flask or FastAPI
# □ Implement document update and deletion functionality
# □ Add support for different file formats (PDF, Word, etc.)
# □ Create performance monitoring and analytics dashboard
# □ Implement user authentication and access control
# □ Add support for multi-modal documents (text + images)
# □ Create automated document ingestion from external sources

