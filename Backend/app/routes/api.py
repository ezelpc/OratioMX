from fastapi import APIRouter, UploadFile, File
from app.models.mobilenet import signLanguageModel

router = APIRouter()
model = signLanguageModel()

@router.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    image_path = f"images/{file.filename}"

    with open(image_path, "wb") as buffer:
        buffer.write(await file.read())
    
    prediction = model.predict(image_path)
    return {"prediction": prediction}
