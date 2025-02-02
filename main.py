from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import google.generativeai as genai
from fastapi.responses import StreamingResponse
import asyncio
# Initialize FastAPI app
app = FastAPI()

# Set Google API Key (Replace with your actual API key)
GOOGLE_API_KEY = "AIzaSyBXyJkhFlfbUPQnttkyxQEKdrhse7HmLGI"  # Replace with actual key
genai.configure(api_key=GOOGLE_API_KEY)

# Pydantic model for /generate/ route
class GenerateRequest(BaseModel):
    user_request: str

@app.post("/generate/")
async def generate_content(request: GenerateRequest):
    try:
        # Initialize Gemini model
        model = genai.GenerativeModel("gemini-pro")

        # Generate content
        response = model.generate_content(request.user_request)
        markdown_content = f"## Generated Content\n\n{response.text}"

        return {"generated_content": markdown_content}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Pydantic model for /summarize/ route
class Article(BaseModel):
    content: str

@app.get("/")
async def root():
    return {"message": "FastAPI server is running!"}

@app.post("/summarize/")
async def summarize(article: Article):
    try:
        # Initialize Gemini model
        model = genai.GenerativeModel("gemini-pro")

        # Format the prompt
        prompt = f"Write a concise summary of the following:\n\n\"{article.content}\"\n\nCONCISE SUMMARY:"

        # Generate response in streaming mode
        response_stream = model.generate_content(prompt, stream=True)

        # Async generator for streaming response
        async def text_stream():
            try:
                for chunk in response_stream:
                    if chunk.text:
                        yield chunk.text + "\n"
                        await asyncio.sleep(0.1)  # Simulate streaming delay
            except Exception as e:
                yield f"Error: {str(e)}"

        return StreamingResponse(text_stream(), media_type="text/plain")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the FastAPI app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
