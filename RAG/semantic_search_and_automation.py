"""
Review Embedding and Semantic Search System

This system helps you find similar customer issues, recommend relevant responses, and identify common themes in feedback. This is semantic search for
product reviews and customer feedback. Find similar content and cluster feedback automatically.

"""

import os
import numpy as np
from typing import Optional
from datetime import datetime
import json
from litellm import embedding
import ollama

class ReviewEmbeddingSystem:
    """
    Manages embeddings for product reviews and customer feedback.
    """

    def __init__(self, model: str = "embeddinggemma"):
        """
        Initialize the review embedding system.
        """
        self.client = ollama.embed
        self.model = model

        # Format: List of dicts with keys: "text", "embedding", "metadata"
        self.embeddings_store = []

    def create_embedding(self, text: str) -> list[float]:
        return (self.client(model=self.model,
                            input=text)
                )["embeddings"][0]

    def embed_review(self, review_text: str, metadata: dict) -> dict:
        data = { 'text':review_text,'embedding':self.create_embedding(review_text),'metadata':metadata}
        self.embeddings_store.append(data)
        return data

    def embed_reviews(self, reviews: list[dict]) -> list[dict]:
        texts = [review["text"] for review in reviews]

        response = self.client(
            model=self.model,
            input=texts,
        )

        embeddings = response["embeddings"]
        for review, embedding_i in zip(reviews, embeddings):
            self.embeddings_store.append({
                "text": review["text"],
                "embedding": embedding_i,
                "metadata": review.get("metadata", {})
            })
        return self.embeddings_store
    
    def calculate_similarity(self, embedding1: list[float], embedding2: list[float]) -> float:
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        return np.dot(vec1,vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2)+1e-8)

    def find_similar_reviews(
        self,
        query: str,
        top_k: int = 5,
        min_similarity: float = 0.5
    ) -> list[tuple[dict, float]]:
        """
        Find reviews most similar to a query using semantic search.
        """
        possible=[]
        query_embed = self.create_embedding(query)
        for x in self.embeddings_store:
            score = self.calculate_similarity(x['embedding'],query_embed)
            if(score>min_similarity):
                possible.append((x,score))
        return sorted(possible,key=lambda x : x[1],reverse=True) [:top_k]

    def find_similar_to_review(
        self,
        review_index: int,
        top_k: int = 5
    ) -> list[tuple[dict, float]]:
        reference_review = self.embeddings_store[review_index]
        reference_embedding = reference_review["embedding"]

        results = []
        for i, review in enumerate(self.embeddings_store):
            if i == review_index:
                continue 

            similarity = self.calculate_similarity(
                reference_embedding,
                review["embedding"]
            )
            results.append((review, similarity))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def cluster_feedback(
        self,
        num_clusters: int = 5,
        method: str = "kmeans"
    ) -> dict[int, list[dict]]:
        if len(self.embeddings_store) == 0:
            return {}
        embeddings = np.array([r["embedding"] for r in self.embeddings_store])
        indices = np.random.choice(len(embeddings), num_clusters, replace=False)
        centroids = embeddings[indices]
        clusters = {i: [] for i in range(num_clusters)}

        for review in self.embeddings_store:
            emb = np.array(review["embedding"])
            distances = [
                1 - self.calculate_similarity(emb.tolist(), c.tolist())
                for c in centroids
            ]
            cluster_id = np.argmin(distances)
            clusters[cluster_id].append(review)

        return clusters

    def save_embeddings(self, filepath: str):
        with open(filepath, "w") as f:
            json.dump(self.embeddings_store, f)

    def load_embeddings(self, filepath: str):
        with open(filepath) as f:
            self.embeddings_store = json.load(f)


# Sample product review dataset for testing
SAMPLE_REVIEWS = [
    {
        "text": "The laptop arrived quickly and works great! Very satisfied with the purchase.",
        "metadata": {"product": "laptop", "rating": 5, "date": "2024-01-15"}
    },
    {
        "text": "Terrible experience. The product broke after 2 days and customer service was unhelpful.",
        "metadata": {"product": "headphones", "rating": 1, "date": "2024-01-16"}
    },
    {
        "text": "Good quality but delivery took too long. Item arrived 2 weeks after expected date.",
        "metadata": {"product": "keyboard", "rating": 3, "date": "2024-01-17"}
    },
    {
        "text": "Amazing sound quality! These headphones are worth every penny. Highly recommend.",
        "metadata": {"product": "headphones", "rating": 5, "date": "2024-01-18"}
    },
    {
        "text": "The product stopped working after a week. Very disappointed with the quality.",
        "metadata": {"product": "mouse", "rating": 1, "date": "2024-01-19"}
    },
    {
        "text": "Fast shipping and excellent packaging. The laptop exceeded my expectations!",
        "metadata": {"product": "laptop", "rating": 5, "date": "2024-01-20"}
    },
    {
        "text": "Not impressed. The keyboard feels cheap and some keys stick occasionally.",
        "metadata": {"product": "keyboard", "rating": 2, "date": "2024-01-21"}
    },
    {
        "text": "Shipping was slow but the product quality is excellent. Would buy again.",
        "metadata": {"product": "monitor", "rating": 4, "date": "2024-01-22"}
    },
    {
        "text": "Customer service was amazing when I had an issue. They resolved it immediately.",
        "metadata": {"product": "laptop", "rating": 5, "date": "2024-01-23"}
    },
    {
        "text": "The item arrived damaged and the return process was complicated.",
        "metadata": {"product": "monitor", "rating": 2, "date": "2024-01-24"}
    }
]


def demonstrate_embedding_creation():
    """
    Demonstrate creating embeddings for reviews.
    """
    print("\n" + "="*70)
    print("DEMO 1: Creating Embeddings")
    print("="*70)
    system = ReviewEmbeddingSystem()

    print("\nCreating embeddings for sample reviews...")
    print(f"Processing {len(SAMPLE_REVIEWS)} reviews...\n")

    embedded_reviews = system.embed_reviews(SAMPLE_REVIEWS)
    
    print(f"Successfully created {len(embedded_reviews)} embeddings")
    print(f"\nExample embedding (first 10 dimensions):")
    print(embedded_reviews[0]["embedding"][:10])
    print(f"Embedding dimension: {len(embedded_reviews[0]['embedding'])}")
    print(f"\nReview text: {embedded_reviews[0]['text'][:80]}...")

    return system


def demonstrate_similarity_search():
    """
    Demonstrate semantic search for similar reviews.
    """
    print("\n" + "="*70)
    print("DEMO 2: Semantic Search")
    print("="*70)

    system = ReviewEmbeddingSystem()
    system.embed_reviews(SAMPLE_REVIEWS)

    # Test queries that use semantic understanding
    queries = [
        "product broke quickly",
        "great customer service",
        "slow delivery"
    ]

    print("\nTesting semantic search with different queries:\n")

    for query in queries:
        print(f"Query: '{query}'")
        results = system.find_similar_reviews(query, top_k=3)
    
        print(f"Found {len(results)} similar reviews:")
        for i, (review, similarity) in enumerate(results, 1):
            print(f"\n  {i}. Similarity: {similarity:.3f}")
            print(f"     Text: {review['text'][:70]}...")
            print(f"     Rating: {review['metadata']['rating']} stars")
        print("\n" + "-"*70)


def demonstrate_similarity_calculation():
    """
    Demonstrate similarity calculation between specific reviews.
    """
    print("\n" + "="*70)
    print("DEMO 3: Review Similarity Calculation")
    print("="*70)

    system = ReviewEmbeddingSystem()
    system.embed_reviews(SAMPLE_REVIEWS)

    print("\nFinding reviews similar to specific examples:\n")

    # Find reviews similar to the first negative review (index 1)
    print("Original review (negative about product breaking):")
    print(f"  {system.embeddings_store[1]['text']}")
    print(f"  Rating: {system.embeddings_store[1]['metadata']['rating']}")
    
    similar = system.find_similar_to_review(1, top_k=3)
    print("\nMost similar reviews:")
    for i, (review, similarity) in enumerate(similar, 1):
        print(f"\n  {i}. Similarity: {similarity:.3f}")
        print(f"     Text: {review['text'][:70]}...")
        print(f"     Rating: {review['metadata']['rating']} stars")


def demonstrate_clustering():
    """
    Demonstrate clustering reviews by topic.
    """
    print("\n" + "="*70)
    print("DEMO 4: Clustering Feedback by Theme")
    print("="*70)

    system = ReviewEmbeddingSystem()
    system.embed_reviews(SAMPLE_REVIEWS)

    print("\nClustering reviews into thematic groups...\n")

    clusters = system.cluster_feedback(num_clusters=3)
    
    for cluster_id, reviews in clusters.items():
        print(f"\nCluster {cluster_id}: {len(reviews)} reviews")
        print("Sample reviews:")
        for review in reviews[:2]:  # Show first 2 from each cluster
            print(f"  - {review['text'][:60]}...")
            print(f"    Rating: {review['metadata']['rating']} stars")


def demonstrate_practical_use_cases():
    """
    Demonstrate practical applications of embeddings in customer service.
    """
    print("\n" + "="*70)
    print("DEMO 5: Practical Customer Service Use Cases")
    print("="*70)


    system = ReviewEmbeddingSystem()
    system.embed_reviews(SAMPLE_REVIEWS)

    print("\nUse Case 1: Finding Similar Customer Issues")
    print("-" * 70)

    new_complaint = "My order hasn't arrived and it's been 3 weeks"
    print(f"New customer complaint: '{new_complaint}'")
    similar_issues = system.find_similar_reviews(new_complaint, top_k=3)
    
    print("\nSimilar past issues:")
    for i, (review, similarity) in enumerate(similar_issues, 1):
        print(f"  {i}. [{similarity:.3f}] {review['text'][:60]}...")

    print("\n\nUse Case 2: Recommending Template Responses")
    print("-" * 70)
    print("Based on similar issues, suggest appropriate response templates")

    print("\n\nUse Case 3: Identifying Trending Issues")
    print("-" * 70)
    print("Cluster recent reviews to identify common problems requiring attention")


def main():
    """
    Run all demonstrations of the embedding system.
    """
    print("\n" + "="*70)
    print("REVIEW EMBEDDING AND SEMANTIC SEARCH SYSTEM")
    print("Customer Service Use Case")
    print("="*70)

    print("\nDemo to show how to:")
    print("1. Create embeddings for customer reviews")
    print("2. Find semantically similar reviews")
    print("3. Calculate similarity between specific reviews")
    print("4. Cluster feedback to identify themes")
    print("5. Apply embeddings to practical customer service scenarios")

    # Run demonstrations

    demonstrate_embedding_creation()
    demonstrate_similarity_search()
    demonstrate_similarity_calculation()
    demonstrate_clustering()
    demonstrate_practical_use_cases()


if __name__ == "__main__":
    main()
