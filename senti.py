from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Initialize the FastAPI app
app = FastAPI()

# Load the BERT uncased model and tokenizer
model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Create a sentiment analysis pipeline
sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# Define the input data structure using Pydantic
class TextsRequest(BaseModel):
    texts: list[str]

# Map star ratings to positive/negative/neutral sentiment
def map_sentiment(label):
    stars = int(label.split()[0])  # Extract star number
    if stars <= 2:
        return "NEGATIVE"
    elif stars >= 4:
        return "POSITIVE"
    else:
        return "NEUTRAL"

# Define the POST request endpoint for sentiment analysis
@app.post("/sentiment-analysis")
async def sentiment_analysis(request: TextsRequest):
    texts = request.texts
    # Perform sentiment analysis
    results = sentiment_pipeline(texts)
    
    # Process and return the results
    response = []
    for text, result in zip(texts, results):
        sentiment = map_sentiment(result['label'])
        response.append({
            "text": text,
            "sentiment": sentiment,
            "confidence": round(result['score'], 2)
        })
    
    return {"results": response}

# To run the application, use the command below:
# uvicorn <filename>:app --reload
