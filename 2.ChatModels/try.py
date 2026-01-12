from langchain_huggingface import HuggingFaceEndpoint
# from dotenv import load_dotenv

# load_dotenv()

model = HuggingFaceEndpoint(
            repo_id= "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            task="text-generation",
            max_new_tokens=512,
            huggingfacehub_api_token="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",
            provider="hf-inference"
            
        )
print(model.invoke("What is Deep Learning?"))