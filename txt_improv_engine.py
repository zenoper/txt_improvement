import spacy
from sentence_transformers import SentenceTransformer, util


def analyze_text(text):
    nlp = spacy.load("en_core_web_md") # run script "python -m spacy download en_core_web_md"

    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

    phrases = [
        "Optimal performance",
        "Utilise resources",
        "Enhance productivity",
        "Conduct an analysis",
        "Maintain a high standard",
        "Implement best practices",
        "Ensure compliance",
        "Streamline operations",
        "Foster innovation",
        "Drive growth",
        "Leverage synergies",
        "Demonstrate leadership",
        "Exercise due diligence",
        "Maximize stakeholder value",
        "Prioritise tasks",
        "Facilitate collaboration",
        "Monitor performance metrics",
        "Execute strategies",
        "Gauge effectiveness",
        "Champion change"
    ]

    # Split text into Sentences
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]

    # Calculate similarity
    results = []
    for sentence in sentences:
        similarities = []
        for phrase in phrases:
            embeddings1 = model.encode(sentence)
            embeddings2 = model.encode(phrase)
            similarity = util.cos_sim(embeddings1, embeddings2).item()

            if similarity > 0.3:
                similarities.append((phrase, similarity))

        similarities = sorted(similarities, key=lambda x: x[1], reverse=True)[:2]

        if similarities:
            results.append(f"Sentence: '{sentence}'")
            for phrase, sim in similarities:
                results.append(f"  Best match: '{phrase}' with similarity: {sim:.4f}")
            results.append("")  # For a blank line between sentences

    return "\n".join(results)


text = input("Please enter the text you wish to analyze:\n")

results = analyze_text(text)
print("\nResults:\n")
print(results)
