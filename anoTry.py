import os
from huggingface_hub import InferenceClient

client = InferenceClient(
    api_key="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",
)

completion = client.chat.completions.create(
    model="google/gemma-2-2b-it",
    messages=[
        {
            "role": "user",
            "content": "What is the capital of France?"
        }
    ],
)

print(completion.choices[0].message.content)