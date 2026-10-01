from transformers import pipeline

classifier = pipeline("sentiment-analysis")

sentences = [
    "I really like this application.",
    "This movie is amazing.",
    "I hate this product.",
    "The service is very disappointing."
]

for sentence in sentences:
    result = classifier(sentence)

    print("Sentence:", sentence)
    print("Result:", result)
    print()
