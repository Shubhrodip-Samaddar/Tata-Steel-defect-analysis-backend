from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
from predict import predict_defect

app = FastAPI(title="Defect Analysis API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "https://tata-steel-defect-analysis-frontend.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "Defect Analysis API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    defect, confidence, all_probs = predict_defect(image)
    return {
        "defect": defect,
        "confidence": round(confidence, 2),
        "all_probabilities": all_probs,
        "filename": file.filename
    }