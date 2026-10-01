```python
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch

# Load model dan tokenizer
model_name = "distilbert-base-cased-distilled-squad"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

# Context
context = '''
Politeknik Elektronika Negeri Surabaya is a university
in Indonesia. PENS focuses on electronics,
engineering, information technology, and multimedia.
The campus is located in Surabaya.
'''

# Questions
questions = [
    "Where is PENS located?",
    "What does PENS focus on?",
    "What is the name of the university?"
]

# Question Answering
for question in questions:

    # Tokenisasi pertanyaan dan context
    inputs = tokenizer(
        question,
        context,
        return_tensors="pt"
    )

    # Prediksi model
    with torch.no_grad():
        outputs = model(**inputs)

    # Menentukan posisi awal dan akhir jawaban
    start_index = torch.argmax(outputs.start_logits, dim=-1).item()
    end_index = torch.argmax(outputs.end_logits, dim=-1).item()

    # Pastikan posisi akhir tidak lebih kecil dari posisi awal
    if end_index < start_index:
        end_index = start_index

    # Mengambil jawaban
    answer = tokenizer.decode(
        inputs["input_ids"][0][start_index:end_index + 1],
        skip_special_tokens=True
    )

    # Menghitung score
    start_probs = torch.softmax(outputs.start_logits, dim=-1)
    end_probs = torch.softmax(outputs.end_logits, dim=-1)

    start_score = start_probs[0, start_index]
    end_score = end_probs[0, end_index]

    score = ((start_score + end_score) / 2).item()

    # Menampilkan hasil
    print("Question:", question)
    print("Answer:", answer)
    print("Score:", score)
    print()
```
