# import sys
# from fastapi import FastAPI, File, UploadFile, Form, HTTPException
# from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
# from fastapi.middleware.cors import CORSMiddleware
# from pathlib import Path
# import shutil
# import uuid
# import asyncio
# import traceback
# # 'Optional' is no longer needed
# # from typing import Optional 

# app = FastAPI(title="AI Video Guide Generator API (Demo Mode)")

# # --- Configuration ---
# BASE_DIR = Path(__file__).resolve().parent
# UPLOAD_DIR = BASE_DIR / "uploads"
# OUTPUT_DIR = BASE_DIR / "outputs"
# FRONTEND_DIR = BASE_DIR / "frontend"

# # Create necessary directories
# UPLOAD_DIR.mkdir(exist_ok=True)
# OUTPUT_DIR.mkdir(exist_ok=True)
# FRONTEND_DIR.mkdir(exist_ok=True)

# # --- Middleware ---
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # --- In-memory Store ---
# processing_status = {}

# # --- Helper Function for DEMO Background Processing ---
# async def find_demo_video(request_id: str, original_filename: str):
#     """
#     This is the new "demo" function. It doesn't run any pipeline.
#     It just finds the pre-generated video that matches the uploaded filename.
#     """
#     try:
#         # 1. Get the base name (e.g., "screen_recording1")
#         base_name = Path(original_filename).stem
#         print(f"[{request_id}] DEMO MODE: Looking for pre-generated video for '{base_name}'")
        
#         # 2. Construct the expected path
#         final_video_path = OUTPUT_DIR / base_name / f"{base_name}_final_guide.mp4"

#         # 3. Simulate a 4-minute processing time
#         print(f"[{request_id}] Simulating 4-minute processing time...")
#         await asyncio.sleep(240) 

#         # 4. Check if the demo file actually exists
#         if not final_video_path.is_file():
#             print(f"[{request_id}] FAILED: Demo video not found at {final_video_path}")
#             processing_status[request_id].update({
#                 "status": "failed",
#                 "error": f"Demo video '{final_video_path.name}' not found. Make sure it is pre-generated in the 'outputs/{base_name}/' folder."
#             })
#             return

#         # 5. If it exists, mark as 'completed' and provide the path
#         print(f"[{request_id}] SUCCESS: Found demo video at {final_video_path}")
#         processing_status[request_id].update({
#             "status": "completed",
#             "output_path": str(final_video_path)
#         })

#     except Exception as e:
#         print(f"[{request_id}] FAILED: An unexpected error occurred: {e}")
#         traceback.print_exc()
#         processing_status[request_id].update({"status": "failed", "error": str(e)})

# # --- API Endpoints ---
# @app.get("/", response_class=HTMLResponse)
# async def serve_frontend():
#     index_path = FRONTEND_DIR / "index.html"
#     if not index_path.is_file():
#         return HTMLResponse("<h1>Frontend/index.html not found</h1>", status_code=404)
#     return FileResponse(index_path)

# @app.post("/api/generate-video")
# async def generate_video(
#     video: UploadFile = File(...),
#     prompt: str = Form(...)
#     # --- FIX ---
#     # 'video_type' parameter has been completely removed.
#     # --- END FIX ---
# ):
#     request_id = str(uuid.uuid4())
    
#     # We still save the uploaded file, but we won't process it.
#     video_path = UPLOAD_DIR / f"{request_id}_{video.filename}"
#     try:
#         with open(video_path, "wb") as buffer:
#             shutil.copyfileobj(video.file, buffer)
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {e}")

#     # Set initial status
#     # --- FIX ---
#     # Removed 'video_type' from the status dictionary
#     processing_status[request_id] = {"status": "processing", "prompt": prompt}
#     # --- END FIX ---
    
#     # Call the "find_demo_video" function
#     asyncio.create_task(find_demo_video(request_id, video.filename))

#     return JSONResponse({
#         "request_id": request_id,
#         "status": "processing",
#         "message": "Video generation started (Demo Mode).",
#         "status_url": f"/api/status/{request_id}"
#     })

# @app.get("/api/video/{request_id}")
# async def get_video(request_id: str):
#     if request_id not in processing_status:
#         raise HTTPException(status_code=404, detail="Video request not found.")
    
#     info = processing_status[request_id]
    
#     if info["status"] != "completed":
#         error_detail = info.get("error", f"Current status: {info['status']}")
#         raise HTTPException(status_code=400, detail=f"Video not ready. {error_detail}")
        
#     path = Path(info.get("output_path"))
    
#     if not path or not path.exists():
#         raise HTTPException(status_code=404, detail="Demo video file is missing from the server.")
        
#     return FileResponse(path, media_type="video/mp4", filename=f"generated_video_{request_id}.mp4")

# @app.get("/api/status/{request_id}")
# async def get_status(request_id: str):
#     if request_id not in processing_status:
#         raise HTTPException(status_code=404, detail="Request ID not found.")
#     return processing_status[request_id]

# @app.delete("/api/video/{request_id}")
# async def delete_video(request_id: str):
#     if request_id not in processing_status:
#         raise HTTPException(status_code=404, detail="Video request not found.")
    
#     # Delete the uploaded video
#     for f in UPLOAD_DIR.glob(f"{request_id}_*"):
#         if f.is_file():
#             f.unlink()
            
#     # We NO LONGER delete the output file
#     del processing_status[request_id]
#     return {"message": "Request data and uploaded file deleted successfully."}

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)
import sys
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import shutil
import uuid
import asyncio
import traceback

# --- NEW IMPORT ---
from fastapi.concurrency import run_in_threadpool

# --- Import the real pipeline ---
try:
    from run_pipeline import run_full_pipeline
except ImportError:
    print("❌ Critical Error: 'run_pipeline.py' not found.")
    print("Make sure 'run_pipeline.py' is in the same directory as this 'main.py' file.")
    sys.exit(1)


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
# (Note: This is fine for a demo, but for production with multiple workers,
# you'd replace this with Redis or a database)

# --- Helper Function for ACTUAL Background Processing ---
async def process_video_pipeline(request_id: str, video_path: Path, prompt: str, voice: str):
    """
    Runs the full video generation pipeline in a separate thread pool.
    Updates the processing_status dictionary with success or failure.
    """
    try:
        print(f"[{request_id}] Starting video processing for: {video_path.name}")
        
        # --- THIS IS THE CRITICAL FIX ---
        # Run the blocking function in a thread pool to avoid freezing the server
        await run_in_threadpool(run_full_pipeline, str(video_path), prompt, voice)
        # -----------------------------
        
        # 2. If successful, determine the output path
        video_name = video_path.stem
        final_video_path = OUTPUT_DIR / video_name / f"{video_name}_final_guide.mp4"

        if not final_video_path.is_file():
            print(f"[{request_id}] ERROR: Pipeline finished but output file not found at {final_video_path}")
            raise FileNotFoundError(f"Pipeline completed but final video was not created at {final_video_path}.")

        # 3. Update status to 'completed'
        processing_status[request_id].update({
            "status": "completed",
            "output_path": str(final_video_path)
        })
        print(f"[{request_id}] Video processing COMPLETED. Output: {final_video_path}")

    except Exception as e:
        # 4. If failed, update status to 'failed'
        print(f"[{request_id}] Video processing FAILED. Error: {e}")
        traceback.print_exc() # Log the full error to the console
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
    voice: str = Form("nova")
):
    request_id = str(uuid.uuid4())
    # Save uploaded video with a unique name to avoid conflicts
    video_path = UPLOAD_DIR / f"{request_id}_{video.filename}"
    try:
        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {e}")

    # Set initial status (video_type removed)
    processing_status[request_id] = {"status": "processing", "prompt": prompt, "voice": voice}

    # Run the REAL pipeline in the background
    asyncio.create_task(process_video_pipeline(request_id, video_path, prompt, voice))

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
    
    # --- FIXED FILE DELETION LOGIC ---
    # Delete the uploaded video
    # Reconstruct the unique upload path
    info = processing_status[request_id]
    original_stem = Path(info.get("prompt", "video")).stem # Failsafe
    
    # Find the uploaded file (we don't know the suffix, so glob)
    for f in UPLOAD_DIR.glob(f"*_{request_id}.*"):
        if f.is_file():
            print(f"Deleting uploaded file: {f}")
            f.unlink()
            break
            
    # Delete the generated video and its directory
    if info.get("status") == "completed":
        output_path = Path(info.get("output_path"))
        if output_path.exists():
            # The output path is .../outputs/STEM_REQUEST_ID/STEM_REQUEST_ID_final_guide.mp4
            # We need to delete the parent directory
            output_dir = output_path.parent
            if output_dir.exists():
                print(f"Deleting output directory: {output_dir}")
                shutil.rmtree(output_dir, ignore_errors=True)
    # ---------------------------------

    del processing_status[request_id]
    return {"message": "Request data and associated files deleted successfully."}

if __name__ == "__main__":
    import uvicorn
    print("--- Starting AI Video Guide Server ---")
    print("Access the frontend at: http://127.0.0.1:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)