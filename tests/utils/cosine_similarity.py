def compare_semantic_similarity(first_text: str, second_text: str) -> float:
    from sentence_transformers import SentenceTransformer, util

    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Compute embeddings
    embeddings1 = model.encode(first_text, convert_to_tensor=True)
    embeddings2 = model.encode(second_text, convert_to_tensor=True)

    # Compute cosine similarity
    cosine_similarity = util.pytorch_cos_sim(embeddings1, embeddings2)

    return cosine_similarity.item()
