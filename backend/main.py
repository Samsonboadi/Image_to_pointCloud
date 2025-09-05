# backend/main.py - Updated FastAPI with SPAR3D Integration

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
import asyncio
import json
import logging
import os
import tempfile
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
import aiofiles
from PIL import Image
import io

# Import processors
from models.spar3d_processor import SPAR3DProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AI Point Cloud Generator API v2.0 - SPAR3D Edition", version="2.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create output directories
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# Global processors
spar3d_processor: Optional[SPAR3DProcessor] = None

# Job storage
jobs: Dict[str, Dict[str, Any]] = {}

class JobStatus:
    PENDING = "pending"
    PROCESSING = "processing" 
    COMPLETED = "completed"
    ERROR = "error"

@app.on_event("startup")
async def startup_event():
    """Initialize processors on startup"""
    global spar3d_processor
    
    logger.info("🚀 Starting AI Point Cloud Generator API v2.0 - SPAR3D Edition")
    
    # Initialize SPAR3D processor
    try:
        spar3d_processor = SPAR3DProcessor(low_vram_mode=True)
        await spar3d_processor.load_model()
        logger.info("✅ SPAR3D processor ready")
    except Exception as e:
        logger.error(f"❌ Failed to initialize SPAR3D: {e}")
        logger.info("💡 Make sure SPAR3D is installed: pip install git+https://github.com/Stability-AI/stable-point-aware-3d.git")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    if spar3d_processor:
        await spar3d_processor.cleanup()
    logger.info("🛑 API shutdown complete")

@app.get("/")
async def root():
    """API health check"""
    return {
        "message": "AI Point Cloud Generator API v2.0 - SPAR3D Edition",
        "status": "ready",
        "models": {
            "spar3d": spar3d_processor is not None,
        },
        "features": [
            "Professional 3D mesh generation",
            "Textured GLB export", 
            "Point cloud generation",
            "UV mapping",
            "Material properties",
            "Real-time preview"
        ],
        "timestamp": datetime.now().isoformat()
    }

@app.get("/models")
async def get_available_models():
    """Get list of available models and their capabilities"""
    models = []
    
    # SPAR3D model
    if spar3d_processor:
        models.append({
            "id": "spar3d",
            "name": "SPAR3D",
            "type": "image_to_3d",
            "description": "Professional textured 3D mesh generation from single images",
            "capabilities": [
                "textured_mesh", 
                "point_cloud", 
                "uv_mapping", 
                "materials", 
                "normal_maps",
                "pbr_materials"
            ],
            "speed": "very_fast",
            "quality": "professional",
            "vram_required": "7GB (low) / 10.5GB (normal)",
            "available": True,
            "outputs": ["glb", "ply"],
            "features": [
                "Sub-second generation",
                "UV-unwrapped textures",
                "Material properties (roughness, metallic)",
                "Backside reconstruction",
                "Point cloud conditioning"
            ],
            "license": "Enterprise license required for commercial use > $1M revenue",
            "recommended": True
        })
    
    return {"models": models, "total": len(models)}

async def process_spar3d_generation(
    job_id: str,
    image: Image.Image,
    settings: Dict[str, Any]
) -> Dict[str, Any]:
    """Process SPAR3D 3D mesh generation"""
    
    if not spar3d_processor:
        raise RuntimeError("SPAR3D processor not available")
    
    # Update job progress
    jobs[job_id]["progress"] = 10
    jobs[job_id]["message"] = "Preprocessing image for SPAR3D..."
    
    # Generate 3D mesh
    result = await spar3d_processor.generate_3d_mesh(
        image=image,
        texture_resolution=settings.get("texture_resolution", 1024),
        guidance_scale=settings.get("guidance_scale", 3.0),
        seed=settings.get("seed"),
        remove_background=settings.get("remove_background", True),
        foreground_ratio=settings.get("foreground_ratio", 1.3),
        remesh_option=settings.get("remesh_option", "none"),
        target_count=settings.get("target_count", 2000)
    )
    
    jobs[job_id]["progress"] = 70
    jobs[job_id]["message"] = "Saving professional 3D assets..."
    
    # Save outputs
    output_dir = OUTPUT_DIR / job_id
    output_dir.mkdir(exist_ok=True)
    
    # Save GLB mesh
    glb_path = output_dir / "mesh.glb"
    async with aiofiles.open(glb_path, "wb") as f:
        await f.write(result["mesh_data"])
    
    # Save PLY point cloud if available
    ply_path = None
    if result.get("point_cloud_data"):
        ply_path = output_dir / "pointcloud.ply"
        async with aiofiles.open(ply_path, "wb") as f:
            await f.write(result["point_cloud_data"])
    
    # Save metadata
    metadata_path = output_dir / "metadata.json"
    async with aiofiles.open(metadata_path, "w") as f:
        await f.write(json.dumps(result["metadata"], indent=2))
    
    return {
        "mesh": {
            "vertices": result["metadata"]["vertex_count"],
            "faces": result["metadata"]["face_count"], 
            "has_textures": result["metadata"]["has_textures"],
            "format": "GLB",
            "generation_time": result["metadata"]["generation_time"]
        },
        "downloadUrl": f"/download/{job_id}/mesh.glb",
        "pointCloudUrl": f"/download/{job_id}/pointcloud.ply" if ply_path else None,
        "metadataUrl": f"/download/{job_id}/metadata.json",
        "preview": result.get("preview_data", {}),
        "meshPreview": result.get("preview_data", {}).get("mesh", {}),
        "metadata": result["metadata"]
    }

@app.post("/process")
async def process_image(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    model: str = Form("spar3d"),
    output_format: str = Form("glb"),
    # SPAR3D specific parameters
    texture_resolution: int = Form(1024),
    guidance_scale: float = Form(3.0),
    seed: Optional[int] = Form(None),
    remove_background: bool = Form(True),
    foreground_ratio: float = Form(1.3),
    remesh_option: str = Form("none"),
    target_count: int = Form(2000)
):
    """Process uploaded image with SPAR3D"""
    
    # Validate model
    if model != "spar3d":
        raise HTTPException(status_code=400, detail=f"Model '{model}' not supported. Only 'spar3d' is available.")
    
    if not spar3d_processor:
        raise HTTPException(status_code=503, detail="SPAR3D processor not available. Please check server logs.")
    
    # Generate job ID
    job_id = str(uuid.uuid4())
    
    # Initialize job
    jobs[job_id] = {
        "status": JobStatus.PENDING,
        "progress": 0,
        "message": "Starting professional 3D generation...",
        "created_at": datetime.now().isoformat(),
        "model": model,
        "results": None
    }
    
    # Load and validate image
    try:
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        # Validate image size
        if image.size[0] * image.size[1] > 4096 * 4096:
            raise ValueError("Image too large. Maximum resolution: 4096x4096")
            
    except Exception as e:
        jobs[job_id]["status"] = JobStatus.ERROR
        jobs[job_id]["message"] = f"Invalid image: {str(e)}"
        raise HTTPException(status_code=400, detail=f"Invalid image: {str(e)}")
    
    # Prepare settings
    settings = {
        "output_format": output_format,
        "texture_resolution": min(max(texture_resolution, 512), 2048),  # Clamp to valid range
        "guidance_scale": max(1.0, min(guidance_scale, 10.0)),  # Clamp to valid range
        "seed": seed,
        "remove_background": remove_background,
        "foreground_ratio": max(1.0, min(foreground_ratio, 2.0)),  # Clamp to valid range
        "remesh_option": remesh_option,
        "target_count": max(100, min(target_count, 20000))  # Clamp to valid range
    }
    
    # Start background processing
    async def process_job():
        try:
            jobs[job_id]["status"] = JobStatus.PROCESSING
            jobs[job_id]["progress"] = 5
            jobs[job_id]["message"] = "Initializing SPAR3D..."
            
            # Process with SPAR3D
            results = await process_spar3d_generation(job_id, image, settings)
            
            jobs[job_id]["status"] = JobStatus.COMPLETED
            jobs[job_id]["progress"] = 100
            jobs[job_id]["message"] = "Professional 3D generation completed successfully!"
            jobs[job_id]["results"] = results
            
            logger.info(f"✅ Job {job_id} completed successfully")
            
        except Exception as e:
            logger.error(f"❌ Job {job_id} failed: {e}")
            jobs[job_id]["status"] = JobStatus.ERROR
            jobs[job_id]["message"] = str(e)
    
    # Start processing in background
    background_tasks.add_task(process_job)
    
    return {
        "job_id": job_id, 
        "status": "started",
        "message": "SPAR3D generation started",
        "estimated_time": "< 10 seconds"
    }

@app.get("/status/{job_id}")
async def get_job_status(job_id: str):
    """Get job processing status"""
    
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    
    response = {
        "job_id": job_id,
        "status": job["status"],
        "progress": job["progress"],
        "message": job["message"],
        "created_at": job["created_at"],
        "model": job["model"]
    }
    
    if job["status"] == JobStatus.COMPLETED and job["results"]:
        response["results"] = job["results"]
    
    return response

@app.get("/download/{job_id}/{filename}")
async def download_file(job_id: str, filename: str):
    """Download generated files"""
    
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    file_path = OUTPUT_DIR / job_id / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    # Determine media type
    media_type = "application/octet-stream"
    if filename.endswith(".glb"):
        media_type = "model/gltf-binary"
    elif filename.endswith(".ply"):
        media_type = "application/ply"
    elif filename.endswith(".json"):
        media_type = "application/json"
    elif filename.endswith(".png"):
        media_type = "image/png"
    elif filename.endswith(".jpg") or filename.endswith(".jpeg"):
        media_type = "image/jpeg"
    
    return FileResponse(
        path=file_path,
        media_type=media_type,
        filename=filename
    )

@app.get("/jobs")
async def list_jobs(limit: int = 10, status: Optional[str] = None):
    """List recent jobs"""
    
    filtered_jobs = []
    for job_id, job_data in jobs.items():
        if status is None or job_data["status"] == status:
            filtered_jobs.append({
                "job_id": job_id,
                "status": job_data["status"],
                "progress": job_data["progress"],
                "created_at": job_data["created_at"],
                "model": job_data["model"]
            })
    
    # Sort by creation time (newest first)
    filtered_jobs.sort(key=lambda x: x["created_at"], reverse=True)
    
    return {
        "jobs": filtered_jobs[:limit],
        "total": len(filtered_jobs)
    }

@app.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    """Delete a job and its outputs"""
    
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Delete job data
    del jobs[job_id]
    
    # Delete output files
    output_dir = OUTPUT_DIR / job_id
    if output_dir.exists():
        import shutil
        shutil.rmtree(output_dir)
    
    return {"message": f"Job {job_id} deleted successfully"}

# Serve static files
app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")

@app.get("/health")
async def health_check():
    """Detailed health check"""
    
    health_status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "spar3d": spar3d_processor is not None,
        },
        "system": {
            "active_jobs": len([j for j in jobs.values() if j["status"] == JobStatus.PROCESSING]),
            "total_jobs": len(jobs),
            "output_dir_exists": OUTPUT_DIR.exists()
        }
    }
    
    # Check GPU if available
    try:
        import torch
        if torch.cuda.is_available():
            health_status["gpu"] = {
                "available": True,
                "device_count": torch.cuda.device_count(),
                "device_name": torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else None,
                "memory_allocated": f"{torch.cuda.memory_allocated(0) / 1e9:.1f}GB" if torch.cuda.device_count() > 0 else None
            }
        else:
            health_status["gpu"] = {"available": False}
    except ImportError:
        health_status["gpu"] = {"error": "PyTorch not available"}
    
    return health_status

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        log_level="info"
    )