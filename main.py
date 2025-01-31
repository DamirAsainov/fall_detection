from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import StreamingResponse, HTMLResponse
import cv2
import uvicorn
from ml.fall_detector import FallDetector



app = FastAPI()
fall_detector = FallDetector()

def generate_video(source):
    cap = cv2.VideoCapture(source)
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        processed_frame = fall_detector.process_frame(frame)
        _, buffer = cv2.imencode(".jpg", processed_frame)
        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")
    cap.release()

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    file_path = f"uploads/{file.filename}"
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())
    return {"success": True, "file_path": file_path}


@app.get("/stream")
async def stream_video(request: Request):
    camera_url = request.query_params.get("camera_url")
    if not camera_url:
        return {"error": "camera_url is required"}

    print(camera_url)
    return StreamingResponse(generate_video(camera_url), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/")
def main_page():
    return HTMLResponse(open("static/index.html").read())

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
