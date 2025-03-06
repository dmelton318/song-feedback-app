from fastapi import FastAPI, File, UploadFile
import librosa
import numpy as np
import soundfile as sf
import uvicorn
import tempfile
import os

app = FastAPI()

def analyze_audio(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None)
        
        # Extract features and ensure they are converted to Python floats
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        tempo = float(tempo)

        spectral_centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
        spectral_bandwidth = float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)))
        rmse = float(np.mean(librosa.feature.rms(y=y)))

        feedback = {
            "tempo": f"Estimated BPM: {tempo:.2f}",
            "spectral_centroid": f"Brightness of the sound: {spectral_centroid:.2f}",
            "spectral_bandwidth": f"Frequency spread: {spectral_bandwidth:.2f}",
            "rmse": f"Dynamic range (RMS energy): {rmse:.2f}",
            "comments": "Consider adjusting the tempo for better groove and balancing brightness for clarity."
        }

        return feedback

    except Exception as e:
        return {"error": str(e)}


import shutil
import asyncio

@app.post("/upload/")
async def upload_audio(file: UploadFile = File(...)):
    try:
        # Save file to a temporary directory (Render limits /tmp usage)
        temp_file_path = f"/tmp/{file.filename}"
        
        # Stream file to disk instead of loading it all into memory
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Process file asynchronously
        feedback = await analyze_audio_async(temp_file_path)

        # Clean up file
        os.remove(temp_file_path)

        return {"filename": file.filename, "feedback": feedback}

    except Exception as e:
        return {"error": str(e)}

# Convert analyze_audio() to async function
async def analyze_audio_async(file_path):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, analyze_audio, file_path)


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port, timeout_keep_alive=120)

