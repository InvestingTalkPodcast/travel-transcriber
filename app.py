from fastapi import FastAPI

app = FastAPI(title="Travel Transcriber")

@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "travel-transcriber"
    }
