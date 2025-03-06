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


@app.post("/upload/")
async def upload_audio(file: UploadFile = File(...)):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            temp_file.write(await file.read())
            temp_file_path = temp_file.name
        feedback = analyze_audio(temp_file_path)
        os.remove(temp_file_path)
        return {"filename": file.filename, "feedback": feedback}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 10000))  # Render dynamically assigns a port
    uvicorn.run(app, host="0.0.0.0", port=port)

