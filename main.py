# # main.py
# from fastapi import FastAPI, File, UploadFile, Form, HTTPException
# from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
# from fastapi.middleware.cors import CORSMiddleware
# from pathlib import Path
# import subprocess
# import shutil
# import uuid

# app = FastAPI(title="AI Video Guide Generator API")

# # Enable CORS for frontend
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # replace with your frontend URL in production
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# UPLOAD_DIR = Path("uploads")
# OUTPUT_DIR = Path("outputs")
# UPLOAD_DIR.mkdir(exist_ok=True)
# OUTPUT_DIR.mkdir(exist_ok=True)

# # Store processing status
# processing_status = {}

# # Serve the HTML UI
# @app.get("/", response_class=HTMLResponse)
# async def root():
#     index_path = Path("frontend/index.html")
#     if not index_path.exists():
#         return HTMLResponse("<h1>Frontend not found</h1>", status_code=404)
#     return index_path.read_text(encoding="utf-8")


# @app.post("/api/generate-video")
# async def generate_video(
#     video: UploadFile = File(...),
#     prompt: str = Form(...),
#     video_type: str = Form(...)
# ):
#     request_id = str(uuid.uuid4())

#     try:
#         # Save uploaded video
#         video_path = UPLOAD_DIR / f"{request_id}_{video.filename}"
#         with open(video_path, "wb") as buffer:
#             shutil.copyfileobj(video.file, buffer)

#         # Output path
#         output_path = OUTPUT_DIR / f"{request_id}_output.mp4"

#         # Update status
#         processing_status[request_id] = {"status": "processing", "video_type": video_type, "prompt": prompt}

#         # Determine command
#         if video_type == "direct":
#             # Direct workflow: video_path + prompt
#             command = ["python", "run_pipeline_direct.py", str(video_path), prompt]

#         else:  # creative
#             # Project name = uploaded video filename without extension
#             project_name = video_path.stem
#             project_folder = UPLOAD_DIR / project_name
#             project_folder.mkdir(exist_ok=True)
#             shutil.move(str(video_path), project_folder / video_path.name)

#             # Assemble video using project_name
#             command = ["python", "assemble_video.py", project_name]

#         # Run subprocess with utf-8 decoding
#         result = subprocess.run(
#             command,
#             stdout=subprocess.PIPE,
#             stderr=subprocess.PIPE,
#             text=True,
#             encoding="utf-8",
#             errors="ignore",
#             timeout=600
#         )

#         if result.returncode != 0:
#             processing_status[request_id]["status"] = "failed"
#             processing_status[request_id]["error"] = result.stderr
#             raise HTTPException(status_code=500, detail=f"Video generation failed: {result.stderr}")

#         # Determine the final video path
#         if video_type == "direct":
#             final_video_path = video_path.with_name(f"{request_id}_output.mp4")
#         else:
#             # assemble_video.py typically outputs inside outputs/ or project folder
#             final_video_path = OUTPUT_DIR / f"{request_id}_output.mp4"
#             if not final_video_path.exists():
#                 # fallback: check inside project folder
#                 final_video_path = project_folder / "output.mp4"
#                 if not final_video_path.exists():
#                     raise HTTPException(status_code=500, detail="Output video not found after creative workflow")

#         processing_status[request_id]["status"] = "completed"
#         processing_status[request_id]["output_path"] = str(final_video_path)

#         return JSONResponse({
#             "request_id": request_id,
#             "status": "completed",
#             "message": "Video generated successfully",
#             "video_url": f"/api/video/{request_id}"
#         })

#     except subprocess.TimeoutExpired:
#         processing_status[request_id]["status"] = "failed"
#         processing_status[request_id]["error"] = "Processing timeout"
#         raise HTTPException(status_code=504, detail="Video processing timeout")
#     except Exception as e:
#         processing_status[request_id]["status"] = "failed"
#         processing_status[request_id]["error"] = str(e)
#         if video_path.exists():
#             video_path.unlink()
#         raise HTTPException(status_code=500, detail=str(e))


# @app.get("/api/video/{request_id}")
# async def get_video(request_id: str):
#     if request_id not in processing_status:
#         raise HTTPException(status_code=404, detail="Video not found")
#     info = processing_status[request_id]
#     if info["status"] != "completed":
#         raise HTTPException(status_code=400, detail=f"Video not ready: {info['status']}")
#     path = Path(info["output_path"])
#     if not path.exists():
#         raise HTTPException(status_code=404, detail="Video file missing")
#     return FileResponse(path, media_type="video/mp4", filename=f"guide_video_{request_id}.mp4")


# @app.get("/api/status/{request_id}")
# async def get_status(request_id: str):
#     if request_id not in processing_status:
#         raise HTTPException(status_code=404, detail="Request not found")
#     return processing_status[request_id]


# @app.delete("/api/video/{request_id}")
# async def delete_video(request_id: str):
#     if request_id not in processing_status:
#         raise HTTPException(status_code=404, detail="Video not found")
#     info = processing_status[request_id]
#     if "output_path" in info:
#         path = Path(info["output_path"])
#         if path.exists():
#             path.unlink()
#     del processing_status[request_id]
#     return {"message": "Video deleted successfully"}


# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import shutil
import uuid
import asyncio

app = FastAPI(title="AI Video Guide Generator API")

# --- Configuration ---
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
FRONTEND_DIR = BASE_DIR / "frontend"

# Create necessary directories
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
FRONTEND_DIR.mkdir(exist_ok=True)

# --- Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- In-memory Store ---
processing_status = {}

# --- Helper Function for Background Simulation ---
async def simulate_video_processing(request_id: str, video_type: str):
    try:
        final_video_name = ""
        if video_type == "direct":
            await asyncio.sleep(120)  # Wait for 3 minutes (180 seconds)
            final_video_name = "screen_recording2_direct_final_video.mp4"
        else:  # creative
            await asyncio.sleep(120)  # Wait for 5 seconds
            final_video_name = "screen_recording2_creative_final_video.mp4"

        final_video_path = OUTPUT_DIR / final_video_name

        if not final_video_path.is_file():
            processing_status[request_id].update({
                "status": "failed",
                "error": f"Required video '{final_video_name}' not found in the 'outputs' folder."
            })
            return

        processing_status[request_id].update({
            "status": "completed",
            "output_path": str(final_video_path)
        })

    except Exception as e:
        processing_status[request_id].update({"status": "failed", "error": str(e)})

# --- API Endpoints ---
@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    index_path = FRONTEND_DIR / "index.html"
    if not index_path.is_file():
        return HTMLResponse("<h1>Frontend/index.html not found</h1>", status_code=404)
    return FileResponse(index_path)

@app.post("/api/generate-video")
async def generate_video(
    video: UploadFile = File(...),
    prompt: str = Form(...),
    video_type: str = Form(...)
):
    request_id = str(uuid.uuid4())
    video_path = UPLOAD_DIR / f"{request_id}_{video.filename}"
    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(video.file, buffer)

    processing_status[request_id] = {"status": "processing", "video_type": video_type, "prompt": prompt}
    asyncio.create_task(simulate_video_processing(request_id, video_type))

    return JSONResponse({
        "request_id": request_id,
        "status": "processing",
        "message": "Video generation started.",
        "status_url": f"/api/status/{request_id}"
    })

@app.get("/api/video/{request_id}")
async def get_video(request_id: str):
    if request_id not in processing_status:
        raise HTTPException(status_code=404, detail="Video request not found.")
    info = processing_status[request_id]
    if info["status"] != "completed":
        error_detail = info.get("error", f"Current status: {info['status']}")
        raise HTTPException(status_code=400, detail=f"Video not ready. {error_detail}")
    path = Path(info.get("output_path"))
    if not path or not path.exists():
        raise HTTPException(status_code=404, detail="Video file is missing from the server.")
    return FileResponse(path, media_type="video/mp4", filename=f"generated_video_{request_id}.mp4")

@app.get("/api/status/{request_id}")
async def get_status(request_id: str):
    if request_id not in processing_status:
        raise HTTPException(status_code=404, detail="Request ID not found.")
    return processing_status[request_id]

@app.delete("/api/video/{request_id}")
async def delete_video(request_id: str):
    if request_id not in processing_status:
        raise HTTPException(status_code=404, detail="Video request not found.")
    for f in UPLOAD_DIR.glob(f"{request_id}_*"):
        if f.is_file():
            f.unlink()
    del processing_status[request_id]
    return {"message": "Request data and uploaded file deleted successfully."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)