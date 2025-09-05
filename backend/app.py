# backend/app.py
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import numpy as np
import cv2
from PIL import Image
import io
import base64
import json
from transformers import pipeline, AutoImageProcessor, AutoModelForDepthEstimation
import trimesh
import open3d as o3d
import laspy
from pathlib import Path
import uuid
from typing import Optional, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Image to Point Cloud API", version="1.0.0")

# CORS middleware for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model storage
models_cache = {}
processing_jobs = {}

# Limits to keep memory reasonable with very large inputs
MAX_IMAGE_DIM = 2048  # max width/height used for processing
DEPTH_PREVIEW_MAX = 2048  # cap preview image size

class ProcessingRequest(BaseModel):
    model: str = "triposr"
    output_format: str = "las"
    point_density: str = "medium"
    coordinate_system: str = "WGS84"
    gps_coords: Optional[Dict[str, float]] = None
    invert_depth: bool = True  # DA-v2 predicts inverse depth; invert for intuitive z
    depth_scale: float = 10.0  # scales normalized depth to world units
    smooth_depth: bool = False  # optional smoothing to reduce speckle
    smooth_ksize: int = 5       # kernel size for Gaussian blur (odd)

class ProcessingStatus(BaseModel):
    job_id: str
    status: str  # pending, processing, completed, error
    progress: int  # 0-100
    message: str
    results: Optional[Dict[str, Any]] = None

def load_model(model_name: str):
    """Load and cache AI models"""
    if model_name in models_cache:
        return models_cache[model_name]
    
    logger.info(f"Loading model: {model_name}")
    
    try:
        if model_name == "triposr":
            # TripoSR integration would require the actual model files
            # For demo purposes, we'll simulate this
            model = {"type": "triposr", "loaded": True}
            
        elif model_name == "depth-anything-v2":
            # Load Depth Anything V2 from Hugging Face
            processor = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
            model = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
            model = {"processor": processor, "model": model, "type": "depth"}
            
        elif model_name == "instantmesh":
            # InstantMesh integration
            model = {"type": "instantmesh", "loaded": True}
            
        else:
            raise ValueError(f"Unsupported model: {model_name}")
            
        models_cache[model_name] = model
        logger.info(f"Model {model_name} loaded successfully")
        return model
        
    except Exception as e:
        logger.error(f"Error loading model {model_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

def process_with_depth_anything(image: np.ndarray, model_info: dict) -> np.ndarray:
    """Process image with Depth Anything V2"""
    try:
        # Convert numpy array to PIL Image
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Process with model
        processor = model_info["processor"]
        model = model_info["model"]
        
        inputs = processor(images=pil_image, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model(**inputs)
            predicted_depth = outputs.predicted_depth
        
        # Convert to numpy (keep raw model output; handle normalization later)
        depth = predicted_depth.squeeze().cpu().numpy().astype(np.float32)
        
        return depth
        
    except Exception as e:
        logger.error(f"Error in depth estimation: {str(e)}")
        raise

def depth_to_point_cloud(image: np.ndarray, depth: np.ndarray,
                        density: str = "medium",
                        invert: bool = True,
                        depth_scale: float = 10.0,
                        smooth: bool = False,
                        smooth_ksize: int = 5,
                        fov: Optional[float] = None) -> np.ndarray:
    """Convert depth map to 3D point cloud.

    Ensures the depth map matches the image resolution to avoid OOB indexing.
    """
    try:
        img_h, img_w = image.shape[:2]
        dep_h, dep_w = depth.shape[:2]

        # Resize depth to image size if needed
        if (dep_h, dep_w) != (img_h, img_w):
            depth = cv2.resize(depth, (img_w, img_h), interpolation=cv2.INTER_LINEAR)

        # Robust normalization with percentile clipping to reduce outliers
        d = depth.astype(np.float32)
        # replace non-finite with median
        finite_mask = np.isfinite(d)
        if not np.all(finite_mask):
            med = np.nanmedian(d)
            d = np.where(finite_mask, d, med)
        p2, p98 = np.percentile(d, [2, 98])
        if p98 <= p2:
            p2, p98 = float(d.min()), float(d.max())
        if p98 > p2:
            d = np.clip(d, p2, p98)
            d = (d - p2) / (p98 - p2 + 1e-6)
        else:
            d = np.zeros_like(d)
        if invert:
            d = 1.0 - d

        # Optional smoothing
        if smooth:
            k = max(3, int(smooth_ksize) // 2 * 2 + 1)  # make odd >=3
            try:
                d = cv2.GaussianBlur(d, (k, k), 0)
            except Exception:
                pass

        h, w = img_h, img_w

        # Camera intrinsics (estimated)
        # If FOV provided (horizontal, degrees), compute focal length in pixels
        cx, cy = w / 2.0, h / 2.0
        if fov and fov > 0:
            f = (w / 2.0) / np.tan(np.deg2rad(fov) / 2.0)
        else:
            f = max(w, h) * 1.2

        # Sampling based on density
        step = {"low": 4, "medium": 2, "high": 1}[density]

        points = []
        colors = []

        for v in range(0, h, step):
            for u in range(0, w, step):
                z = float(d[v, u]) * float(depth_scale)
                # Allow zero/near-zero depths to pass so preview is never empty
                x = (u - cx) * (z if z != 0.0 else 1e-6) / f
                y = (v - cy) * (z if z != 0.0 else 1e-6) / f

                points.append([x, y, z])

                # Get color from original image (BGR -> RGB)
                if image.ndim == 3 and image.shape[2] >= 3:
                    b, g, r = image[v, u][:3]
                    colors.append([int(r), int(g), int(b)])
                else:
                    colors.append([128, 128, 128])

        return np.array(points, dtype=np.float32), np.array(colors, dtype=np.float32)

    except Exception as e:
        logger.error(f"Error in point cloud generation: {str(e)}")
        raise


def refine_point_cloud(points: np.ndarray, colors: np.ndarray,
                       nb_neighbors: int = 20, std_ratio: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    """Denoise point cloud using statistical outlier removal.

    Returns filtered points and colors, preserving alignment.
    """
    try:
        if points is None or len(points) == 0:
            return points, colors
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
        if colors is not None and len(colors) == len(points):
            pcd.colors = o3d.utility.Vector3dVector(np.clip(colors / 255.0, 0, 1))
        _, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
        ind = np.asarray(ind, dtype=np.int64)
        points_f = points[ind]
        colors_f = colors[ind] if colors is not None and len(colors) == len(points) else colors
        return points_f, colors_f
    except Exception as e:
        logger.warning(f"Point cloud refinement failed: {e}")
        return points, colors
    
def save_mesh_from_points(points: np.ndarray, colors: np.ndarray, filename: str,
                          method: str = "poisson") -> tuple[str, o3d.geometry.TriangleMesh]:
    """Create a surface mesh from a point cloud and save as PLY.

    method: 'poisson' (default) or 'bpa' (ball pivoting)
    """
    Path("outputs").mkdir(exist_ok=True)
    filepath = f"outputs/{filename}.ply"

    # Build Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    if colors is not None and len(colors) == len(points):
        pcd.colors = o3d.utility.Vector3dVector(np.clip(colors / 255.0, 0, 1))

    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=30))

    if method == "bpa":
        # Estimate radius based on average spacing
        distances = pcd.compute_nearest_neighbor_distance()
        if len(distances) == 0:
            raise ValueError("Not enough points for meshing")
        avg_dist = float(np.mean(distances))
        radii = [avg_dist * 1.5, avg_dist * 2.0, avg_dist * 2.5]
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd, o3d.utility.DoubleVector(radii)
        )
    else:
        # Poisson reconstruction
        mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8)
        # Crop to the bounding box of points to remove far artifacts
        bbox = pcd.get_axis_aligned_bounding_box()
        mesh = mesh.crop(bbox)

    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()

    o3d.io.write_triangle_mesh(filepath, mesh)
    return filepath, mesh



def save_point_cloud(points: np.ndarray, colors: np.ndarray, 
                     format: str, filename: str) -> str:
    """Save point cloud in various formats"""
    try:
        if format.lower() == "ply":
            return save_ply(points, colors, filename)
        elif format.lower() in ["las", "laz"]:
            return save_las(points, colors, filename)
        elif format.lower() == "xyz":
            return save_xyz(points, colors, filename)
        else:
            raise ValueError(f"Unsupported format: {format}")
            
    except Exception as e:
        logger.error(f"Error saving point cloud: {str(e)}")
        raise

def save_ply(points: np.ndarray, colors: np.ndarray, filename: str) -> str:
    """Save as PLY format"""
    filepath = f"outputs/{filename}.ply"
    Path("outputs").mkdir(exist_ok=True)
    
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)
    
    # Save
    o3d.io.write_point_cloud(filepath, pcd)
    return filepath

def save_las(points: np.ndarray, colors: np.ndarray, filename: str) -> str:
    """Save as LAS format for GIS compatibility (laspy v2)."""
    filepath = f"outputs/{filename}.las"
    Path("outputs").mkdir(exist_ok=True)

    # Create LAS header: point_format 2 supports RGB; version 1.2 is widely compatible
    header = laspy.LasHeader(point_format=2, version="1.2")

    # Set scaling and offsets (LAS stores scaled integers internally)
    scale = 0.01
    offset = [float(points[:, 0].min()), float(points[:, 1].min()), float(points[:, 2].min())]
    header.scales = np.array([scale, scale, scale], dtype=np.float64)
    header.offsets = np.array(offset, dtype=np.float64)

    if points is None or len(points) == 0:
        raise ValueError("No points to write to LAS")

    # Create LAS data and assign coordinates (LasData handles scaling)
    las = laspy.LasData(header)
    las.x = points[:, 0]
    las.y = points[:, 1]
    las.z = points[:, 2]

    # Optional colors (convert 0-255 to 0-65535)
    if colors is not None and len(colors) == len(points):
        c = np.clip(colors, 0, 255).astype(np.uint16)
        las.red = (c[:, 0] * 256)
        las.green = (c[:, 1] * 256)
        las.blue = (c[:, 2] * 256)
    else:
        # Provide default mid-gray if colors missing but format supports RGB
        las.red = np.full(len(points), 32768, dtype=np.uint16)
        las.green = np.full(len(points), 32768, dtype=np.uint16)
        las.blue = np.full(len(points), 32768, dtype=np.uint16)

    las.write(filepath)
    return filepath

def save_xyz(points: np.ndarray, colors: np.ndarray, filename: str) -> str:
    """Save as XYZ ASCII format"""
    filepath = f"outputs/{filename}.xyz"
    Path("outputs").mkdir(exist_ok=True)
    
    with open(filepath, 'w') as f:
        for i in range(len(points)):
            x, y, z = points[i]
            r, g, b = colors[i] if len(colors) > 0 else [128, 128, 128]
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")
    
    return filepath

def generate_gis_metadata(points: np.ndarray, request: ProcessingRequest) -> dict:
    """Generate GIS-compatible metadata"""
    bounds = {
        "minX": float(points[:, 0].min()),
        "maxX": float(points[:, 0].max()),
        "minY": float(points[:, 1].min()),
        "maxY": float(points[:, 1].max()),
        "minZ": float(points[:, 2].min()),
        "maxZ": float(points[:, 2].max()),
    }
    
    metadata = {
        "coordinateSystem": request.coordinate_system,
        "bounds": bounds,
        "pointCount": len(points),
        "generatedWith": request.model,
        "outputFormat": request.output_format,
        "pointDensity": request.point_density,
    }
    
    if request.gps_coords:
        metadata["gpsReference"] = request.gps_coords
    
    return metadata

async def process_image_pipeline(job_id: str, image_data: bytes, request: ProcessingRequest):
    """Main processing pipeline"""
    try:
        processing_jobs[job_id]["status"] = "processing"
        processing_jobs[job_id]["progress"] = 10
        processing_jobs[job_id]["message"] = "Loading AI model..."
        
        # Load the specified model
        model_info = load_model(request.model)
        
        processing_jobs[job_id]["progress"] = 20
        processing_jobs[job_id]["message"] = "Processing image..."
        
        # Convert bytes to image array
        image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Failed to decode image data")

        # Downscale very large images to keep processing and preview stable
        ih, iw = image.shape[:2]
        max_dim = max(ih, iw)
        if max_dim > MAX_IMAGE_DIM:
            scale = MAX_IMAGE_DIM / float(max_dim)
            new_w = int(round(iw * scale))
            new_h = int(round(ih * scale))
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            logger.info(f"Resized input image from {iw}x{ih} to {new_w}x{new_h} for processing")
        
        depth_data_url = None
        if request.model == "depth-anything-v2":
            # Use Depth Anything V2 for depth estimation
            processing_jobs[job_id]["progress"] = 40
            processing_jobs[job_id]["message"] = "Estimating depth..."
            
            depth = process_with_depth_anything(image, model_info)
            # Encode depth to a grayscale PNG data URL for frontend preview
            try:
                d = depth.astype(np.float32)
                # Robust normalization (percentile clip) similar to point-cloud path
                finite = np.isfinite(d)
                if not np.any(finite):
                    raise ValueError("Depth has no finite values for preview")
                p2, p98 = np.percentile(d[finite], [2, 98])
                if p98 <= p2:
                    p2, p98 = float(d.min()), float(d.max())
                if p98 > p2:
                    d = np.clip(d, p2, p98)
                    d = (d - p2) / (p98 - p2 + 1e-6)
                else:
                    d = np.zeros_like(d)
                # Invert for a more intuitive visualization if requested
                d = 1.0 - d if True else d  # uses default invert for preview

                depth_img = (d * 255.0).astype(np.uint8)
                # Ensure preview isn't excessively large
                dh, dw = depth_img.shape[:2]
                dmax = max(dh, dw)
                if dmax > DEPTH_PREVIEW_MAX:
                    s = DEPTH_PREVIEW_MAX / float(dmax)
                    depth_img = cv2.resize(depth_img, (int(round(dw * s)), int(round(dh * s))), interpolation=cv2.INTER_AREA)
                ok, buf = cv2.imencode('.png', depth_img)
                if ok:
                    depth_b64 = base64.b64encode(buf.tobytes()).decode('utf-8')
                    depth_data_url = f"data:image/png;base64,{depth_b64}"
            except Exception as e:
                logger.warning(f"Failed to encode depth preview: {e}")
            
            processing_jobs[job_id]["progress"] = 60
            processing_jobs[job_id]["message"] = "Generating point cloud..."
            
            points, colors = depth_to_point_cloud(
                image,
                depth,
                density=request.point_density,
                invert=request.invert_depth,
                depth_scale=request.depth_scale,
                smooth=request.smooth_depth,
                fov=getattr(request, 'fov', None),
            )
            # Denoise/refine
            points, colors = refine_point_cloud(points, colors)
            
        else:
            # For other models (TripoSR, InstantMesh, etc.)
            # This would require their specific implementations
            processing_jobs[job_id]["progress"] = 40
            processing_jobs[job_id]["message"] = f"Processing with {request.model}..."
            
            # Simulate processing for demo
            import time
            time.sleep(2)
            
            # Generate dummy point cloud for demo
            points, colors = generate_dummy_point_cloud(image, request.point_density)
        
        processing_jobs[job_id]["progress"] = 80
        processing_jobs[job_id]["message"] = "Saving point cloud..."
        
        # Downsample preview points for frontend viewer (max ~20k)
        try:
            max_preview = 20000
            if len(points) > max_preview:
                stride = max(1, len(points) // max_preview)
                pprev = points[::stride]
                cprev = colors[::stride] if colors is not None and len(colors) else np.zeros_like(pprev)
            else:
                pprev = points
                cprev = colors if colors is not None and len(colors) else np.zeros_like(points)
            # Convert to lists to make JSON serializable
            preview_points = pprev.astype(float).tolist()
            preview_colors = cprev.astype(float).tolist()
        except Exception:
            preview_points, preview_colors = [], []

        # Save artifact based on desired format; also build mesh preview if mesh generated
        mesh_formats = {"mesh_ply", "mesh"}
        mesh_preview = None
        if request.output_format.lower() in mesh_formats:
            filepath, mesh = save_mesh_from_points(points, colors, job_id, method="poisson")
            # Simplify for preview and extract buffers
            try:
                target_tris = 20000
                mesh_s = mesh.simplify_quadric_decimation(target_tris)
            except Exception:
                mesh_s = mesh
            try:
                v = np.asarray(mesh_s.vertices, dtype=np.float32)
                f = np.asarray(mesh_s.triangles, dtype=np.int32)
                n = np.asarray(mesh_s.vertex_normals, dtype=np.float32) if mesh_s.has_vertex_normals() else np.zeros_like(v)
                # Colors: if no colors on mesh, fill with mid gray
                if mesh_s.has_vertex_colors():
                    c = np.asarray(mesh_s.vertex_colors, dtype=np.float32)
                else:
                    c = np.full_like(v, 0.7, dtype=np.float32)
                mesh_preview = {
                    "vertices": v.astype(float).tolist(),
                    "normals": n.astype(float).tolist(),
                    "colors": (c[:, :3]).astype(float).tolist(),
                    "faces": f.reshape(-1).astype(int).tolist(),
                }
            except Exception as e:
                logger.warning(f"Failed to build mesh preview: {e}")
        else:
            filepath = save_point_cloud(points, colors, request.output_format, job_id)
        
        # Generate GIS metadata
        metadata = generate_gis_metadata(points, request)
        
        processing_jobs[job_id]["progress"] = 100
        processing_jobs[job_id]["status"] = "completed"
        processing_jobs[job_id]["message"] = "Processing complete!"
        processing_jobs[job_id]["results"] = {
            "pointCloud": {
                "filepath": filepath,
                "points": len(points),
                "format": request.output_format.upper(),
            },
            "gisData": metadata,
            "downloadUrl": f"/download/{job_id}",
            "preview": {
                "points": preview_points,
                "colors": preview_colors,
            },
            "meshPreview": mesh_preview,
            "depthMap": depth_data_url
        }
        
    except Exception as e:
        logger.error(f"Error in processing pipeline: {str(e)}")
        processing_jobs[job_id]["status"] = "error"
        processing_jobs[job_id]["message"] = f"Error: {str(e)}"

def generate_dummy_point_cloud(image: np.ndarray, density: str) -> tuple:
    """Generate dummy point cloud for unsupported models (demo purposes)"""
    h, w = image.shape[:2]
    step = {"low": 8, "medium": 4, "high": 2}[density]
    
    points = []
    colors = []
    
    for v in range(0, h, step):
        for u in range(0, w, step):
            # Simple depth based on image intensity
            gray = cv2.cvtColor(image[v:v+1, u:u+1], cv2.COLOR_BGR2GRAY)[0, 0]
            z = (255 - gray) / 255.0 * 5  # Depth from 0-5 units
            
            x = (u - w/2) / 100.0
            y = (v - h/2) / 100.0
            
            points.append([x, y, z])
            colors.append([image[v, u, 2], image[v, u, 1], image[v, u, 0]])  # BGR to RGB
    
    return np.array(points), np.array(colors)

@app.post("/process", response_model=dict)
async def process_image(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    model: str = "depth-anything-v2",
    output_format: str = "las",
    point_density: str = "medium",
    coordinate_system: str = "WGS84"
):
    """Main endpoint for image processing"""
    
    # Validate input
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Generate job ID
    job_id = str(uuid.uuid4())
    
    # Initialize job status
    processing_jobs[job_id] = {
        "status": "pending",
        "progress": 0,
        "message": "Job queued",
        "results": None
    }
    
    # Read image data
    image_data = await file.read()
    
    # Create processing request
    request = ProcessingRequest(
        model=model,
        output_format=output_format,
        point_density=point_density,
        coordinate_system=coordinate_system
    )
    
    # Start background processing
    background_tasks.add_task(process_image_pipeline, job_id, image_data, request)
    
    return {"job_id": job_id, "status": "queued"}

@app.get("/status/{job_id}", response_model=ProcessingStatus)
async def get_job_status(job_id: str):
    """Get processing job status"""
    if job_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job_data = processing_jobs[job_id]
    return ProcessingStatus(
        job_id=job_id,
        status=job_data["status"],
        progress=job_data["progress"],
        message=job_data["message"],
        results=job_data["results"]
    )

@app.get("/download/{job_id}")
async def download_result(job_id: str):
    """Download processed point cloud file"""
    if job_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job_data = processing_jobs[job_id]
    if job_data["status"] != "completed":
        raise HTTPException(status_code=400, detail="Job not completed")
    
    filepath = job_data["results"]["pointCloud"]["filepath"]
    
    from fastapi.responses import FileResponse
    return FileResponse(filepath, media_type="application/octet-stream")

@app.get("/models")
async def list_available_models():
    """List available AI models"""
    models = [
        {
            "id": "triposr",
            "name": "TripoSR",
            "description": "Fast mesh generation (1-2 seconds)",
            "license": "MIT",
            "recommended": True,
            "supported": False  # Requires additional setup
        },
        {
            "id": "instantmesh", 
            "name": "InstantMesh",
            "description": "High quality 3D assets (~10 seconds)",
            "license": "Custom",
            "supported": False  # Requires additional setup
        },
        {
            "id": "depth-anything-v2",
            "name": "Depth Anything V2", 
            "description": "Superior depth estimation + point cloud",
            "license": "Apache-2.0",
            "recommended": True,
            "supported": True
        },
        {
            "id": "spar3d",
            "name": "SPAR3D",
            "description": "Ultra-fast generation (<1 second)", 
            "license": "Community",
            "supported": False  # Requires additional setup
        }
    ]
    
    return {"models": models}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "models_loaded": list(models_cache.keys())}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
