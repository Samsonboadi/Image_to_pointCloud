# backend/models/spar3d_processor.py
import os
import torch
import tempfile
import time
import base64
import logging
from PIL import Image
import numpy as np
from typing import Optional, Dict, Any, Tuple
import trimesh
from trimesh.exchange import gltf
import asyncio
from concurrent.futures import ThreadPoolExecutor

try:
    from spar3d.system import SPAR3D
    from spar3d.utils import foreground_crop, remove_background
    from transparent_background import Remover
    SPAR3D_AVAILABLE = True
except ImportError:
    SPAR3D_AVAILABLE = False
    logging.warning("SPAR3D not available. Install with: pip install git+https://github.com/Stability-AI/stable-point-aware-3d.git")

class SPAR3DProcessor:
    """
    High-performance SPAR3D processor for generating 3D meshes from single images.
    Features:
    - Professional textured mesh generation
    - Point cloud conditioning for better backside reconstruction
    - UV-unwrapped outputs with material properties
    - Optimized memory management
    """
    
    def __init__(self, device: str = "auto", low_vram_mode: bool = False):
        self.device = self._get_device(device)
        self.low_vram_mode = low_vram_mode
        self.model: Optional[SPAR3D] = None
        self.bg_remover: Optional[Remover] = None
        self.executor = ThreadPoolExecutor(max_workers=2)
        
        # SPAR3D Configuration
        self.COND_WIDTH = 512
        self.COND_HEIGHT = 512
        self.COND_DISTANCE = 2.2
        self.COND_FOVY = 0.591627
        self.BACKGROUND_COLOR = [0.5, 0.5, 0.5]
        
        logging.info(f"🚀 SPAR3D Processor initialized on {self.device}")
        if low_vram_mode:
            logging.info("💾 Low VRAM mode enabled (7GB vs 10.5GB)")
    
    def _get_device(self, device: str) -> str:
        """Smart device detection with fallbacks"""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return device
    
    async def load_model(self) -> bool:
        """Load SPAR3D model asynchronously"""
        if not SPAR3D_AVAILABLE:
            raise ImportError("SPAR3D is not installed. Please install it first.")
        
        try:
            logging.info("🔄 Loading SPAR3D model...")
            
            # Load model in thread to avoid blocking
            def _load():
                model = SPAR3D.from_pretrained(
                    "stabilityai/stable-point-aware-3d",
                    config_name="config.yaml",
                    weight_name="model.safetensors",
                    low_vram_mode=self.low_vram_mode
                )
                model.eval()
                model = model.to(self.device)
                return model
            
            loop = asyncio.get_event_loop()
            self.model = await loop.run_in_executor(self.executor, _load)
            
            # Initialize background remover
            self.bg_remover = Remover()
            
            logging.info("✅ SPAR3D model loaded successfully")
            return True
            
        except Exception as e:
            logging.error(f"❌ Failed to load SPAR3D model: {e}")
            return False
    
    def _preprocess_image(self, image: Image.Image, remove_bg: bool = True, foreground_ratio: float = 1.3) -> Image.Image:
        """Preprocess image for SPAR3D inference"""
        # Convert to RGB if needed
        if image.mode != 'RGB':
            if image.mode == 'RGBA':
                # Handle transparency
                background = Image.new('RGB', image.size, (255, 255, 255))
                background.paste(image, mask=image.split()[-1])
                image = background
            else:
                image = image.convert('RGB')
        
        # Remove background if requested
        if remove_bg and self.bg_remover:
            try:
                image_array = np.array(image)
                # Remove background using transparent-background
                result = self.bg_remover.process(image_array, type='map')
                alpha = result.astype(np.float32) / 255.0
                
                # Apply alpha mask
                image_array = image_array.astype(np.float32)
                for c in range(3):
                    image_array[:, :, c] = image_array[:, :, c] * alpha + (1 - alpha) * 255
                
                image = Image.fromarray(image_array.astype(np.uint8))
            except Exception as e:
                logging.warning(f"Background removal failed: {e}. Using original image.")
        
        # Crop foreground
        if foreground_ratio > 1.0:
            try:
                image = foreground_crop(image, foreground_ratio)
            except Exception as e:
                logging.warning(f"Foreground cropping failed: {e}. Using original image.")
        
        # Resize to model input size
        image = image.resize((self.COND_WIDTH, self.COND_HEIGHT), Image.LANCZOS)
        
        return image
    
    async def generate_3d_mesh(
        self,
        image: Image.Image,
        texture_resolution: int = 1024,
        guidance_scale: float = 3.0,
        seed: Optional[int] = None,
        remove_background: bool = True,
        foreground_ratio: float = 1.3,
        remesh_option: str = "none",
        target_count: int = 2000,
        generate_preview: bool = True
    ) -> Dict[str, Any]:
        """
        Generate 3D mesh from single image using SPAR3D
        
        Returns:
            Dict containing:
            - mesh_data: GLB file data (bytes)
            - point_cloud_data: PLY file data (bytes) 
            - preview_data: Preview point cloud for frontend
            - metadata: Generation metadata
        """
        
        if not self.model:
            raise RuntimeError("SPAR3D model not loaded. Call load_model() first.")
        
        start_time = time.time()
        
        # Set seed for reproducibility
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        logging.info(f"🎯 Starting SPAR3D generation with texture_resolution={texture_resolution}")
        
        try:
            # Preprocess image
            processed_image = self._preprocess_image(
                image, remove_background, foreground_ratio
            )
            
            # Run SPAR3D inference in executor to avoid blocking
            def _generate():
                with torch.inference_mode():
                    # Generate mesh using SPAR3D
                    result = self.model.sample(
                        image=processed_image,
                        guidance_scale=guidance_scale,
                        texture_resolution=texture_resolution,
                        remesh=remesh_option,
                        vertex_count=target_count if remesh_option != "none" else None
                    )
                    return result
            
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(self.executor, _generate)
            
            # Extract mesh and point cloud
            mesh = result['mesh']
            point_cloud = result.get('point_cloud', None)
            
            # Export mesh to GLB
            mesh_data = self._export_mesh_to_glb(mesh)
            
            # Export point cloud to PLY
            point_cloud_data = None
            if point_cloud is not None:
                point_cloud_data = self._export_point_cloud_to_ply(point_cloud)
            
            # Generate preview data for frontend
            preview_data = None
            if generate_preview:
                preview_data = self._generate_preview_data(mesh, point_cloud)
            
            generation_time = time.time() - start_time
            
            # Metadata
            metadata = {
                "model": "SPAR3D",
                "generation_time": generation_time,
                "texture_resolution": texture_resolution,
                "guidance_scale": guidance_scale,
                "seed": seed,
                "vertex_count": len(mesh.vertices) if hasattr(mesh, 'vertices') else 0,
                "face_count": len(mesh.faces) if hasattr(mesh, 'faces') else 0,
                "has_textures": hasattr(mesh.visual, 'material'),
                "remesh_option": remesh_option
            }
            
            logging.info(f"✅ SPAR3D generation completed in {generation_time:.2f}s")
            logging.info(f"📊 Generated mesh: {metadata['vertex_count']} vertices, {metadata['face_count']} faces")
            
            return {
                "mesh_data": mesh_data,
                "point_cloud_data": point_cloud_data,
                "preview_data": preview_data,
                "metadata": metadata
            }
            
        except Exception as e:
            logging.error(f"❌ SPAR3D generation failed: {e}")
            raise
    
    def _export_mesh_to_glb(self, mesh: trimesh.Trimesh) -> bytes:
        """Export mesh to GLB format with textures"""
        try:
            # Create scene with mesh
            scene = trimesh.Scene(mesh)
            
            # Export to GLB with all features
            glb_data = gltf.export_glb(
                scene,
                include_normals=True,
                include_textures=True,
                merge_primitives=True
            )
            
            return glb_data
        except Exception as e:
            logging.error(f"Failed to export mesh to GLB: {e}")
            raise
    
    def _export_point_cloud_to_ply(self, point_cloud: np.ndarray) -> bytes:
        """Export point cloud to PLY format"""
        try:
            # Create point cloud mesh
            pc_mesh = trimesh.PointCloud(
                vertices=point_cloud[:, :3],
                colors=point_cloud[:, 3:6] if point_cloud.shape[1] >= 6 else None
            )
            
            # Export to PLY
            ply_data = pc_mesh.export(file_type='ply')
            
            return ply_data.encode() if isinstance(ply_data, str) else ply_data
        except Exception as e:
            logging.error(f"Failed to export point cloud to PLY: {e}")
            raise
    
    def _generate_preview_data(self, mesh: trimesh.Trimesh, point_cloud: Optional[np.ndarray]) -> Dict[str, Any]:
        """Generate preview data for frontend visualization"""
        try:
            preview = {}
            
            # Mesh preview (sample vertices for frontend)
            if hasattr(mesh, 'vertices') and len(mesh.vertices) > 0:
                # Sample vertices for preview (max 5000 for performance)
                max_preview_vertices = 5000
                vertices = mesh.vertices
                
                if len(vertices) > max_preview_vertices:
                    # Random sampling
                    indices = np.random.choice(len(vertices), max_preview_vertices, replace=False)
                    vertices = vertices[indices]
                
                # Get colors if available
                colors = None
                if hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None:
                    colors = mesh.visual.vertex_colors[:len(vertices), :3]  # RGB only
                else:
                    # Default color
                    colors = np.full((len(vertices), 3), [128, 128, 255], dtype=np.uint8)
                
                preview['mesh'] = {
                    'vertices': vertices.tolist(),
                    'colors': colors.tolist(),
                    'faces': mesh.faces.tolist() if hasattr(mesh, 'faces') else [],
                    'normals': mesh.vertex_normals.tolist() if hasattr(mesh, 'vertex_normals') else []
                }
            
            # Point cloud preview
            if point_cloud is not None and len(point_cloud) > 0:
                # Sample points for preview
                max_preview_points = 3000
                points = point_cloud
                
                if len(points) > max_preview_points:
                    indices = np.random.choice(len(points), max_preview_points, replace=False)
                    points = points[indices]
                
                preview['points'] = {
                    'positions': points[:, :3].tolist(),
                    'colors': points[:, 3:6].tolist() if points.shape[1] >= 6 else [[128, 128, 255]] * len(points)
                }
            
            return preview
            
        except Exception as e:
            logging.error(f"Failed to generate preview data: {e}")
            return {}
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.executor:
            self.executor.shutdown(wait=True)
        
        if self.model and hasattr(self.model, 'cpu'):
            self.model.cpu()
        
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        logging.info("🧹 SPAR3D processor cleanup completed")