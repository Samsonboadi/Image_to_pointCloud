import React, { useState, useRef, useCallback, useEffect } from 'react';
import { Upload, Download, Settings, Play, Eye, Map, Loader2, CheckCircle2, RotateCcw, Grid, Box, Layers } from 'lucide-react';
import * as THREE from 'three';
import axios from 'axios';

const App = () => {
  const [uploadedImage, setUploadedImage] = useState(null);
  const [processing, setProcessing] = useState(false);
  const [currentStep, setCurrentStep] = useState(0);
  const [results, setResults] = useState({
    depthMap: null,
    pointCloud: null,
    mesh: null,
    gisData: null,
    jobId: null
  });
  const [settings, setSettings] = useState({
    model: 'depth-anything-v2',
    outputFormat: 'las',
    pointDensity: 'medium',
    coordinateSystem: 'WGS84',
    invertDepth: true,
    depthScale: 15,
    smoothDepth: false,
    fov: 60
  });
  
  // 3D Viewer state
  const [viewerMode, setViewerMode] = useState('pointcloud');
  const [showGrid, setShowGrid] = useState(true);
  const [showAxes, setShowAxes] = useState(true);
  const [pointSize, setPointSize] = useState(3);
  const [wireframe, setWireframe] = useState(false);
  const [viewer3DLoading, setViewer3DLoading] = useState(false);
  const [cameraAutoRotate, setCameraAutoRotate] = useState(false);
  const [showSettingsModal, setShowSettingsModal] = useState(false);
  
  const fileInputRef = useRef(null);
  const mountRef = useRef(null);
  const sceneRef = useRef(null);
  const rendererRef = useRef(null);
  const cameraRef = useRef(null);
  const pointCloudRef = useRef(null);
  const meshRef = useRef(null);
  const controlsRef = useRef({ mouseDown: false, cameraDistance: 8 });
  const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';

  const steps = [
    { name: 'Upload Image', icon: Upload, status: 'pending' },
    { name: 'AI Process', icon: Play, status: 'pending' },
    { name: 'Generate 3D', icon: Box, status: 'pending' },
    { name: 'Export GIS', icon: Map, status: 'pending' }
  ];

  const models = [
    { 
      id: 'triposr', 
      name: 'TripoSR', 
      description: 'Ultra-fast mesh generation',
      license: 'MIT',
      recommended: true,
      speed: '1-2s'
    },
    { 
      id: 'instantmesh', 
      name: 'InstantMesh', 
      description: 'High-quality 3D assets',
      license: 'Custom',
      speed: '~10s'
    },
    { 
      id: 'depth-anything-v2', 
      name: 'Depth Anything V2', 
      description: 'Superior depth â†’ point cloud',
      license: 'Apache-2.0',
      speed: '2-3s'
    },
    { 
      id: 'spar3d', 
      name: 'SPAR3D', 
      description: 'Real-time generation',
      license: 'Community',
      speed: '<1s'
    }
  ];

  // Initialize Three.js scene
  useEffect(() => {
    if (!mountRef.current) return;

    // Scene setup
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0f172a);
    sceneRef.current = scene;

    // Camera setup
    const camera = new THREE.PerspectiveCamera(75, 1, 0.1, 1000);
    camera.position.set(8, 6, 8);
    camera.lookAt(0, 0, 0);
    cameraRef.current = camera;

    // Renderer setup
    const renderer = new THREE.WebGLRenderer({ 
      antialias: true,
      alpha: true,
      powerPreference: "high-performance"
    });
    const cw = mountRef.current.clientWidth || 800;
    const ch = mountRef.current.clientHeight || 600;
    renderer.setSize(cw, ch);
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    mountRef.current.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    // Resize handling (defined later below as well)

    // Lighting setup
    const ambientLight = new THREE.AmbientLight(0x404040, 0.6);
    scene.add(ambientLight);

    const directionalLight = new THREE.DirectionalLight(0xffffff, 1);
    directionalLight.position.set(10, 10, 5);
    directionalLight.castShadow = true;
    scene.add(directionalLight);

    const rimLight = new THREE.DirectionalLight(0x0066ff, 0.3);
    rimLight.position.set(-10, -5, -5);
    scene.add(rimLight);

    // Grid and axes
    if (showGrid) {
      const gridHelper = new THREE.GridHelper(20, 20, 0x444444, 0x222222);
      gridHelper.material.transparent = true;
      gridHelper.material.opacity = 0.5;
      scene.add(gridHelper);
    }

    if (showAxes) {
      const axesHelper = new THREE.AxesHelper(5);
      scene.add(axesHelper);
    }

    // Mouse controls
    const controls = controlsRef.current;
    
    const handleMouseDown = (event) => {
      event.preventDefault();
      controls.mouseDown = true;
      controls.mouseX = event.clientX;
      controls.mouseY = event.clientY;
      renderer.domElement.style.cursor = 'grabbing';
    };

    const handleMouseMove = (event) => {
      if (!controls.mouseDown) return;

      const deltaX = event.clientX - controls.mouseX;
      const deltaY = event.clientY - controls.mouseY;

      const spherical = new THREE.Spherical();
      spherical.setFromVector3(camera.position);
      
      spherical.theta -= deltaX * 0.005;
      spherical.phi += deltaY * 0.005;
      spherical.phi = Math.max(0.1, Math.min(Math.PI - 0.1, spherical.phi));
      spherical.radius = controls.cameraDistance;
      
      camera.position.setFromSpherical(spherical);
      camera.lookAt(0, 0, 0);

      controls.mouseX = event.clientX;
      controls.mouseY = event.clientY;
    };

    const handleMouseUp = () => {
      controls.mouseDown = false;
      renderer.domElement.style.cursor = 'grab';
    };

    const handleWheel = (event) => {
      event.preventDefault();
      const scale = event.deltaY > 0 ? 1.1 : 0.9;
      controls.cameraDistance *= scale;
      controls.cameraDistance = Math.max(2, Math.min(50, controls.cameraDistance));
      
      const direction = camera.position.clone().normalize();
      camera.position.copy(direction.multiplyScalar(controls.cameraDistance));
      camera.lookAt(0, 0, 0);
    };

    renderer.domElement.addEventListener('mousedown', handleMouseDown);
    renderer.domElement.addEventListener('mousemove', handleMouseMove);
    renderer.domElement.addEventListener('mouseup', handleMouseUp);
    renderer.domElement.addEventListener('wheel', handleWheel, { passive: false });
    renderer.domElement.style.cursor = 'grab';

    // Resize handling
    const handleResize = () => {
      if (!mountRef.current) return;
      const width = mountRef.current.clientWidth;
      const height = mountRef.current.clientHeight;
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
      renderer.setSize(width, height);
    };
    
    window.addEventListener('resize', handleResize);
    handleResize();

    // Animation loop
    let animationId;
    const animate = () => {
      animationId = requestAnimationFrame(animate);
      
      if (cameraAutoRotate && !controls.mouseDown) {
        const spherical = new THREE.Spherical();
        spherical.setFromVector3(camera.position);
        spherical.theta += 0.01;
        spherical.radius = controls.cameraDistance;
        camera.position.setFromSpherical(spherical);
        camera.lookAt(0, 0, 0);
      }
      
      renderer.render(scene, camera);
    };
    animate();

    return () => {
      if (animationId) cancelAnimationFrame(animationId);
      if (mountRef.current && renderer.domElement && mountRef.current.contains(renderer.domElement)) {
        mountRef.current.removeChild(renderer.domElement);
      }
      window.removeEventListener('resize', handleResize);
      renderer.dispose();
    };
  }, [showGrid, showAxes, cameraAutoRotate]);

  useEffect(() => {
    if (!sceneRef.current) return;
    // If backend preview is present, render it directly for point clouds
    if (viewerMode === 'pointcloud' && results?.preview?.points?.length) {
      const pts = results.preview.points;
      const cols = results.preview.colors || [];
      const flatPos = new Float32Array(pts.length * 3);
      const flatCol = new Float32Array(pts.length * 3);
      for (let i = 0; i < pts.length; i++) {
        const p = pts[i];
        flatPos[i * 3 + 0] = p[0];
        flatPos[i * 3 + 1] = p[1];
        flatPos[i * 3 + 2] = p[2];
        const c = cols[i] || [128, 128, 128];
        flatCol[i * 3 + 0] = c[0] / 255;
        flatCol[i * 3 + 1] = c[1] / 255;
        flatCol[i * 3 + 2] = c[2] / 255;
      }
      displayPointCloudData({ points: flatPos, colors: flatCol });
      return;
    }
    // If mesh preview is present, render it
    if (viewerMode === 'mesh' && results?.meshPreview?.vertices?.length) {
      const v = results.meshPreview.vertices.flat();
      const n = (results.meshPreview.normals || results.meshPreview.vertices).flat();
      const c = (results.meshPreview.colors || []).flat();
      const f = results.meshPreview.faces || [];
      const vertices = new Float32Array(v);
      const normals = new Float32Array(n);
      const colors = c.length ? new Float32Array(c) : new Float32Array(vertices.length).fill(0.7);
      const faces = new Uint32Array(f);
      displayMeshData({ vertices, normals, colors, faces });
      return;
    }
    // Otherwise try to fetch from backend if we have a job id
    if (results.jobId) {
      load3DDataFromBackend(results.jobId);
      return;
    }
    // Fallback demo
    if (!results.pointCloud && !results.mesh) {
      generateDemo3DContent();
    }
  }, [results.preview, results.pointCloud, results.mesh, results.jobId, pointSize, viewerMode, wireframe]);

  const load3DDataFromBackend = async (jobId) => {
    setViewer3DLoading(true);
    try {
      const res = await fetch(`${API_BASE}/status/${jobId}`);
      if (!res.ok) throw new Error('Status fetch failed');
      const s = await res.json();
      if (s?.results?.preview && viewerMode === 'pointcloud') {
        const pts = s.results.preview.points || [];
        const cols = s.results.preview.colors || [];
        const flatPos = new Float32Array(pts.length * 3);
        const flatCol = new Float32Array(pts.length * 3);
        for (let i = 0; i < pts.length; i++) {
          const p = pts[i];
          flatPos[i*3+0] = p[0];
          flatPos[i*3+1] = p[1];
          flatPos[i*3+2] = p[2];
          const c = cols[i] || [128,128,128];
          flatCol[i*3+0] = c[0]/255;
          flatCol[i*3+1] = c[1]/255;
          flatCol[i*3+2] = c[2]/255;
        }
        displayPointCloudData({ points: flatPos, colors: flatCol });
      } else if (viewerMode === 'mesh') {
        // Mesh preview not provided; show placeholder geometry
        generateDemo3DContent();
      }
    } catch (error) {
      console.error('Error loading 3D data:', error);
      generateDemo3DContent();
    } finally {
      setViewer3DLoading(false);
    }
  };

  const clearScene = () => {
    if (pointCloudRef.current) {
      sceneRef.current.remove(pointCloudRef.current);
      pointCloudRef.current.geometry?.dispose();
      pointCloudRef.current.material?.dispose();
      pointCloudRef.current = null;
    }
    if (meshRef.current) {
      sceneRef.current.remove(meshRef.current);
      meshRef.current.geometry?.dispose();
      meshRef.current.material?.dispose();
      meshRef.current = null;
    }
  };

  const displayPointCloudData = (data) => {
    if (!sceneRef.current) return;
    clearScene();

    const geometry = new THREE.BufferGeometry();
    const positions = new Float32Array(data.points);
    const colors = new Float32Array(data.colors);

    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

    const material = new THREE.PointsMaterial({
      size: Math.max(0.5, pointSize / 15),
      vertexColors: true,
      sizeAttenuation: true
    });

    const points = new THREE.Points(geometry, material);
    pointCloudRef.current = points;
    sceneRef.current.add(points);

    // Fit camera to content
    try {
      geometry.center();
      geometry.computeBoundingSphere();
      const bs = geometry.boundingSphere;
      if (bs && cameraRef.current) {
        const cam = cameraRef.current;
        const radius = Math.max(1, bs.radius);
        cam.position.set(bs.center.x + radius * 2.5, bs.center.y + radius * 1.5, bs.center.z + radius * 2.5);
        cam.lookAt(bs.center);
        controlsRef.current.cameraDistance = radius * 3;
      }
    } catch {}
  };

  const displayMeshData = (data) => {
    if (!sceneRef.current) return;
    clearScene();

    const geometry = new THREE.BufferGeometry();
    const positions = new Float32Array(data.vertices);
    const normals = new Float32Array(data.normals);
    const colors = new Float32Array(data.colors);
    const indices = new Uint16Array(data.faces);

    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute('normal', new THREE.BufferAttribute(normals, 3));
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    geometry.setIndex(new THREE.BufferAttribute(indices, 1));

    const material = new THREE.MeshPhongMaterial({
      vertexColors: true,
      wireframe: wireframe,
      transparent: true,
      opacity: wireframe ? 1.0 : 0.8,
      side: THREE.DoubleSide
    });

    const mesh = new THREE.Mesh(geometry, material);
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    meshRef.current = mesh;
    sceneRef.current.add(mesh);

    // Fit camera to mesh
    try {
      geometry.center();
      geometry.computeBoundingSphere();
      const bs = geometry.boundingSphere;
      if (bs && cameraRef.current) {
        const cam = cameraRef.current;
        const radius = Math.max(1, bs.radius);
        cam.position.set(bs.center.x + radius * 2.5, bs.center.y + radius * 1.5, bs.center.z + radius * 2.5);
        cam.lookAt(bs.center);
        controlsRef.current.cameraDistance = radius * 3;
      }
    } catch {}
  };

  const generateDemo3DContent = () => {
    if (!sceneRef.current) return;
    clearScene();

    if (viewerMode === 'pointcloud') {
      const pointCount = Math.min(results.pointCloud?.points || 15000, 15000);
      const geometry = new THREE.BufferGeometry();
      const positions = new Float32Array(pointCount * 3);
      const colors = new Float32Array(pointCount * 3);

      for (let i = 0; i < pointCount; i++) {
        const i3 = i * 3;
        const x = (Math.random() - 0.5) * 8;
        const z = (Math.random() - 0.5) * 8;
        const y = Math.sin(x * 0.5) * Math.cos(z * 0.5) * 2 + Math.random() * 0.5;
        
        positions[i3] = x;
        positions[i3 + 1] = y;
        positions[i3 + 2] = z;

        const heightNorm = (y + 2) / 4;
        colors[i3] = Math.max(0, heightNorm - 0.3);
        colors[i3 + 1] = Math.max(0, 1 - Math.abs(heightNorm - 0.5) * 2);
        colors[i3 + 2] = Math.max(0, 0.7 - heightNorm);
      }

      geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
      geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

      const material = new THREE.PointsMaterial({
        size: pointSize / 15,
        vertexColors: true,
        sizeAttenuation: true
      });

      const points = new THREE.Points(geometry, material);
      pointCloudRef.current = points;
      sceneRef.current.add(points);
      
    } else if (viewerMode === 'mesh') {
      const geometry = new THREE.SphereGeometry(3, 32, 32);
      
      const positions = geometry.attributes.position;
      for (let i = 0; i < positions.count; i++) {
        const vertex = new THREE.Vector3();
        vertex.fromBufferAttribute(positions, i);
        const noise = (Math.random() - 0.5) * 0.3;
        vertex.multiplyScalar(1 + noise);
        positions.setXYZ(i, vertex.x, vertex.y, vertex.z);
      }
      positions.needsUpdate = true;
      geometry.computeVertexNormals();

      const material = new THREE.MeshPhongMaterial({
        color: wireframe ? 0x00aaff : 0x00ff88,
        wireframe: wireframe,
        transparent: true,
        opacity: wireframe ? 0.8 : 0.9,
        side: THREE.DoubleSide,
        shininess: 100
      });

      const mesh = new THREE.Mesh(geometry, material);
      mesh.castShadow = true;
      mesh.receiveShadow = true;
      meshRef.current = mesh;
      sceneRef.current.add(mesh);
    }
  };

  const handleImageUpload = useCallback((event) => {
    const file = event.target.files[0];
    if (file) {
      if (file.size > 10 * 1024 * 1024) {
        alert('File size must be less than 10MB');
        return;
      }
      
      const reader = new FileReader();
      reader.onload = (e) => {
        setUploadedImage({
          url: e.target.result,
          file: file,
          name: file.name,
          size: (file.size / 1024 / 1024).toFixed(2) + ' MB'
        });
        updateStepStatus(0, 'completed');
      };
      reader.readAsDataURL(file);
    }
  }, []);

  const updateStepStatus = (stepIndex, status) => {
    setCurrentStep(stepIndex);
    steps[stepIndex].status = status;
  };

  const processImage = async () => {
  if (!uploadedImage) return;
  setProcessing(true);
  try {
    updateStepStatus(1, 'processing');
    const form = new FormData();
    form.append('file', uploadedImage.file);
    form.append('model', settings.model);
    form.append('output_format', settings.outputFormat);
    form.append('point_density', settings.pointDensity);
    form.append('coordinate_system', settings.coordinateSystem);

      // Depth and projection tuning
      form.append('invert_depth', String(settings.invertDepth));
      form.append('depth_scale', String(settings.depthScale));
      form.append('smooth_depth', String(settings.smoothDepth));
      form.append('fov', String(settings.fov));

      const startRes = await axios.post(`${API_BASE}/process`, form, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
    const id = startRes.data?.job_id;
    setResults(prev => ({ ...prev, jobId: id }));

    const poll = async () => {
      const res = await axios.get(`${API_BASE}/status/${id}`);
      const s = res.data;
      if (s.progress >= 1 && s.progress < 40) updateStepStatus(1, 'processing');
      if (s.progress >= 40 && s.progress < 80) updateStepStatus(2, 'processing');
      if (s.progress >= 80 && s.progress < 100) updateStepStatus(3, 'processing');
      if (s.status === 'completed') {
        updateStepStatus(1, 'completed');
        updateStepStatus(2, 'completed');
        updateStepStatus(3, 'completed');
        setResults(prev => ({
          ...prev,
          pointCloud: s.results?.pointCloud || null,
          gisData: s.results?.gisData || null,
          depthMap: s.results?.depthMap || null,
          preview: s.results?.preview || null,
          downloadUrl: s.results?.downloadUrl ? `${API_BASE}${s.results.downloadUrl}` : null,
        }));
        setProcessing(false);
        return;
      }
      if (s.status === 'error') {
        console.error('Processing error:', s.message);
        setProcessing(false);
        return;
      }
      setTimeout(poll, 1500);
    };
    poll();
  } catch (e) {
    console.error('Failed to start processing', e);
    setProcessing(false);
  }
};
  const simulateAPICall = (duration) => {
    return new Promise(resolve => setTimeout(resolve, duration));
  };

  const resetCamera = () => {
    if (cameraRef.current) {
      cameraRef.current.position.set(8, 6, 8);
      cameraRef.current.lookAt(0, 0, 0);
      controlsRef.current.cameraDistance = 10;
    }
  };

  const downloadFile = (type) => {
  if (type === 'pointcloud' && results.downloadUrl) {
    window.open(results.downloadUrl, '_blank');
    return;
  }
  if (type === 'mesh' && results.downloadUrl) {
    window.open(results.downloadUrl, '_blank');
    return;
  }
  if (type === 'gis' && results.gisData) {
    const filename = uploadedImage?.name?.replace(/\.[^/.]+$/, '') || 'result';
    const content = JSON.stringify(results.gisData, null, 2);
    const blob = new Blob([content], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${filename}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }
};

  return (
    <div className="h-screen bg-gradient-to-br from-slate-900 to-slate-800 flex overflow-hidden">
      {/* LEFT SIDEBAR - CONFIGURATIONS */}
      <div className="w-80 bg-white shadow-2xl flex flex-col">
        {/* Sidebar Header */}
        <div className="p-6 border-b border-gray-200 bg-gradient-to-r from-blue-50 to-indigo-50">
          <h1 className="text-xl font-bold text-gray-900 mb-1">
            AI Point Cloud Generator
          </h1>
          <p className="text-sm text-gray-600">Image â†’ 3D â†’ GIS Ready</p>
        </div>

        <div className="flex-1 overflow-y-auto">
          
          {/* PROGRESS TRACKER */}
          <div className="p-4 border-b border-gray-100">
            <h3 className="text-sm font-semibold text-gray-700 mb-3 flex items-center">
              <Layers className="w-4 h-4 mr-2 text-blue-500" />
              Processing Steps
            </h3>
            <div className="space-y-2">
              {steps.map((step, index) => {
                const Icon = step.icon;
                const isActive = index === currentStep && processing;
                const isCompleted = step.status === 'completed';
                const isProcessing = step.status === 'processing';
                
                return (
                  <div key={step.name} className={`flex items-center space-x-3 p-2 rounded-lg transition-all ${
                    isActive ? 'bg-blue-50 border border-blue-200' : ''
                  }`}>
                    <div className={`flex items-center justify-center w-7 h-7 rounded-full text-xs font-medium transition-all ${
                      isCompleted 
                        ? 'bg-green-500 text-white shadow-lg' 
                        : isProcessing 
                          ? 'bg-blue-500 text-white animate-pulse shadow-lg' 
                          : 'bg-gray-200 text-gray-400'
                    }`}>
                      {isProcessing ? (
                        <Loader2 className="w-4 h-4 animate-spin" />
                      ) : isCompleted ? (
                        <CheckCircle2 className="w-4 h-4" />
                      ) : (
                        <Icon className="w-4 h-4" />
                      )}
                    </div>
                    <div className="flex-1">
                      <div className={`text-sm font-medium ${
                        isActive ? 'text-blue-700' : isCompleted ? 'text-green-700' : 'text-gray-500'
                      }`}>
                        {step.name}
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* IMAGE UPLOAD SECTION */}
          <div className="p-4 border-b border-gray-100">
            <h3 className="text-sm font-semibold text-gray-700 mb-3 flex items-center">
              <Upload className="w-4 h-4 mr-2 text-green-500" />
              Input Image
            </h3>
            
            {!uploadedImage ? (
              <div
                onClick={() => fileInputRef.current?.click()}
                className="border-2 border-dashed border-gray-300 rounded-xl p-6 text-center cursor-pointer hover:border-blue-400 hover:bg-blue-50 transition-all duration-200 group"
              >
                <Upload className="w-10 h-10 text-gray-400 mx-auto mb-3 group-hover:text-blue-500 transition-colors" />
                <p className="text-sm text-gray-600 mb-1 font-medium">Drop image here</p>
                <p className="text-xs text-gray-400">PNG, JPG â€¢ Max 10MB</p>
              </div>
            ) : (
              <div className="space-y-3">
                <div className="relative">
                  <img
                    src={uploadedImage.url}
                    alt="Uploaded"
                    className="w-full h-36 object-cover rounded-xl shadow-sm"
                  />
                  <div className="absolute top-2 right-2 bg-black bg-opacity-50 text-white px-2 py-1 rounded text-xs">
                    {uploadedImage.size}
                  </div>
                </div>
                <div className="text-xs text-gray-600 bg-gray-50 p-2 rounded-lg">
                  <div className="font-medium truncate">{uploadedImage.name}</div>
                </div>
                <button
                  onClick={() => fileInputRef.current?.click()}
                  className="w-full py-2 text-sm border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors"
                >
                  Change Image
                </button>
              </div>
            )}
            
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              onChange={handleImageUpload}
              className="hidden"
            />
          </div>

        {/* AI MODEL SELECTION (Dropdown) */}
        <label className="block text-sm font-medium text-gray-700 mb-1">
          Model
        </label>
        <select
          onChange={(e) =>
            setSettings((prev) => ({ ...prev, model: e.target.value }))
          }
          className="w-full text-sm border border-gray-300 rounded-lg px-3 py-2 focus:ring-2 focus:ring-blue-500 focus:border-transparent"
        >
          {models.map((m) => (
            <option key={m.name} value={m.name}>
              {m.name} - {m.description}
            </option>
          ))}
        </select>



          {/* GENERATE BUTTON */}
          {uploadedImage && (
            <div className="p-4 border-b border-gray-100">
              <button
                onClick={processImage}
                disabled={processing}
                className="w-full bg-gradient-to-r from-blue-600 to-indigo-600 text-white py-3 px-4 rounded-xl hover:from-blue-700 hover:to-indigo-700 disabled:from-gray-400 disabled:to-gray-500 disabled:cursor-not-allowed transition-all duration-200 flex items-center justify-center gap-2 text-sm font-semibold shadow-lg"
              >
                {processing ? (
                  <>
                    <Loader2 className="w-5 h-5 animate-spin" />
                    Processing...
                  </>
                ) : (
                  <>
                    <Play className="w-5 h-5" />
                    Generate 3D
                  </>
                )}
              </button>
            </div>
          )}

          {/* RESULTS & DOWNLOADS */}
          {results.pointCloud && (
            <div className="p-4">
              <h3 className="text-sm font-semibold text-gray-700 mb-3 flex items-center">
                <Download className="w-4 h-4 mr-2 text-green-500" />
                Download Results
              </h3>
              
              <div className="space-y-3">
                <div className="bg-gradient-to-r from-green-50 to-emerald-50 rounded-lg p-3 border border-green-200">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm font-medium text-green-800">Point Cloud</span>
                    <span className="text-xs text-green-600 bg-green-100 px-2 py-1 rounded">
                      {results.pointCloud.format}
                    </span>
                  </div>
                  <div className="text-xs text-green-700 space-y-1">
                    <div>{results.pointCloud.points.toLocaleString()} points</div>
                    <div>{results.pointCloud.size}</div>
                  </div>
                  <button
                    onClick={() => downloadFile('pointcloud')}
                    className="mt-2 w-full bg-green-600 text-white py-1.5 px-3 rounded-lg text-xs hover:bg-green-700 transition-colors flex items-center justify-center"
                  >
                    <Download className="w-3 h-3 mr-1" />
                    Download {settings.outputFormat.toUpperCase()}
                  </button>
                </div>

                {results.mesh && (
                  <div className="bg-gradient-to-r from-blue-50 to-cyan-50 rounded-lg p-3 border border-blue-200">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-sm font-medium text-blue-800">3D Mesh</span>
                      <span className="text-xs text-blue-600 bg-blue-100 px-2 py-1 rounded">
                        OBJ
                      </span>
                    </div>
                    <div className="text-xs text-blue-700 space-y-1">
                      <div>{results.mesh.vertices.toLocaleString()} vertices</div>
                      <div>{results.mesh.faces.toLocaleString()} faces</div>
                    </div>
                    <button
                      onClick={() => downloadFile('mesh')}
                      className="mt-2 w-full bg-blue-600 text-white py-1.5 px-3 rounded-lg text-xs hover:bg-blue-700 transition-colors flex items-center justify-center"
                    >
                      <Download className="w-3 h-3 mr-1" />
                      Download OBJ
                    </button>
                  </div>
                )}

                {results.gisData && (
                  <div className="bg-gradient-to-r from-purple-50 to-pink-50 rounded-lg p-3 border border-purple-200">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-sm font-medium text-purple-800">GIS Metadata</span>
                      <span className="text-xs text-purple-600 bg-purple-100 px-2 py-1 rounded">
                        JSON
                      </span>
                    </div>
                    <div className="text-xs text-purple-700">
                      <div>{results.gisData.coordinateSystem}</div>
                      <div>Spatial bounds included</div>
                    </div>
                    <button
                      onClick={() => downloadFile('gis')}
                      className="mt-2 w-full bg-purple-600 text-white py-1.5 px-3 rounded-lg text-xs hover:bg-purple-700 transition-colors flex items-center justify-center"
                    >
                      <Download className="w-3 h-3 mr-1" />
                      Download Metadata
                    </button>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      </div>

      {/* RIGHT PANEL - 3D VIEWER */}
      <div className="flex-1 flex flex-col">
        {/* Viewer Header */}
        <div className="bg-white shadow-sm border-b border-gray-200 p-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center">
              <Box className="w-6 h-6 text-indigo-600 mr-3" />
              <div>
                <h2 className="text-lg font-bold text-gray-900">Interactive 3D Viewer</h2>
                <p className="text-sm text-gray-500">
                  {viewerMode === 'pointcloud' ? 'Point Cloud Visualization' : 'Mesh Visualization'}
                </p>
              </div>
            </div>
            
            <div className="flex items-center space-x-3">
              {results.pointCloud && (
                <div className="text-sm text-gray-600 bg-gray-50 px-3 py-1.5 rounded-lg">
                  <span className="font-medium">{viewerMode === 'pointcloud' ? 'Points' : 'Vertices'}:</span>
                  <span className="ml-1 text-indigo-600 font-bold">
                    {(viewerMode === 'pointcloud' ? results.pointCloud?.points : results.mesh?.vertices)?.toLocaleString()}
                  </span>
                </div>
              )}
              {viewerMode === 'pointcloud' && results.preview?.points?.length > 0 && (
                <div className="text-sm text-gray-600 bg-gray-50 px-3 py-1.5 rounded-lg">
                  <span className="font-medium">Preview:</span>
                  <span className="ml-1 text-green-600 font-bold">{results.preview.points.length.toLocaleString()} pts</span>
                </div>
              )}
              {viewerMode === 'mesh' && results.meshPreview?.vertices?.length > 0 && (
                <div className="text-sm text-gray-600 bg-gray-50 px-3 py-1.5 rounded-lg">
                  <span className="font-medium">Preview mesh:</span>
                  <span className="ml-1 text-green-600 font-bold">{results.meshPreview.vertices.length.toLocaleString()} v</span>
                </div>
              )}
              
              <div className="flex items-center space-x-1 bg-gray-50 rounded-lg p-1">
                <button
                  onClick={() => setViewerMode('pointcloud')}
                  className={`p-2 rounded transition-all ${
                    viewerMode === 'pointcloud' 
                      ? 'bg-white shadow-sm text-blue-600' 
                      : 'text-gray-500 hover:text-gray-700'
                  }`}
                  title="Point Cloud View"
                >
                  <Grid className="w-4 h-4" />
                </button>
                <button
                  onClick={() => setViewerMode('mesh')}
                  className={`p-2 rounded transition-all ${
                    viewerMode === 'mesh' 
                      ? 'bg-white shadow-sm text-blue-600' 
                      : 'text-gray-500 hover:text-gray-700'
                  }`}
                  title="Mesh View"
                >
                  <Box className="w-4 h-4" />
                </button>
              </div>
              
              <button
                onClick={resetCamera}
                className="p-2 bg-gray-100 rounded-lg hover:bg-gray-200 transition-colors"
                title="Reset Camera View"
              >
                <RotateCcw className="w-4 h-4" />
              </button>
              <button
                onClick={() => setShowSettingsModal(true)}
                className="p-2 bg-gray-100 rounded-lg hover:bg-gray-200 transition-colors"
                title="Settings"
              >
                <Settings className="w-4 h-4" />
              </button>
            </div>
          </div>
        </div>

        {/* 3D VIEWER MAIN AREA */}
        <div className="flex-1 relative overflow-hidden">
          {viewer3DLoading && (
            <div className="absolute inset-0 bg-slate-900 bg-opacity-90 flex items-center justify-center z-20">
              <div className="text-white text-center">
                <Loader2 className="w-16 h-16 mx-auto mb-4 animate-spin text-blue-400" />
                <p className="text-xl font-semibold mb-2">Loading 3D Data</p>
                <p className="text-sm text-gray-300">
                  Fetching {viewerMode} from AI backend...
                </p>
              </div>
            </div>
          )}
          
          {results.pointCloud || results.mesh ? (
            <div className="relative w-full h-full">
              <div ref={mountRef} className="w-full h-full" />
              
              <div className="absolute top-4 left-4 space-y-2">
                <div className="bg-black bg-opacity-60 text-white p-3 rounded-lg text-sm backdrop-blur-sm">
                  <div className="font-semibold text-blue-300">
                    {viewerMode === 'pointcloud' ? 'Point Cloud' : '3D Mesh'}
                  </div>
                  {viewerMode === 'pointcloud' && results.pointCloud && (
                    <div className="text-gray-200">
                      {results.pointCloud.points.toLocaleString()} points
                    </div>
                  )}
                  {viewerMode === 'mesh' && results.mesh && (
                    <div className="text-gray-200">
                      {results.mesh.vertices.toLocaleString()} vertices<br/>
                      {results.mesh.faces.toLocaleString()} faces
                    </div>
                  )}
                </div>
                
                <div className="bg-black bg-opacity-40 text-white p-2 rounded text-xs backdrop-blur-sm">
                  <div>ðŸ–±ï¸ Drag: Rotate</div>
                  <div>ðŸŽ¯ Wheel: Zoom</div>
                  {cameraAutoRotate && <div>ðŸ”„ Auto-rotating</div>}
                </div>
              </div>
              
              <div className="absolute top-4 right-4 bg-black bg-opacity-60 text-white p-3 rounded-lg text-xs backdrop-blur-sm">
                <div className="text-green-400 font-semibold">â— WebGL Ready</div>
                <div>Three.js Renderer</div>
                <div>Model: {settings.model}</div>
                {!viewer3DLoading && <div className="text-blue-400">Interactive Mode</div>}
              </div>
            </div>
          ) : (
            <div className="flex items-center justify-center h-full text-center">
              <div className="text-gray-300 max-w-md">
                <div className="relative mb-6">
                  <Box className="w-24 h-24 mx-auto mb-4 text-indigo-400 animate-pulse" />
                  <div className="absolute inset-0 bg-indigo-500 opacity-20 rounded-full blur-xl" />
                </div>
                <h3 className="text-2xl font-bold mb-3 text-white">Professional 3D Viewer</h3>
                <p className="text-gray-400 mb-6">Upload an image and generate 3D data to see it visualized here</p>
                
                <div className="bg-slate-800 rounded-xl p-4 text-left space-y-3">
                  <div className="text-sm font-semibold text-white mb-2">âœ¨ Features:</div>
                  <div className="space-y-1 text-xs text-gray-300">
                    <div>ðŸŽ® Interactive mouse controls</div>
                    <div>ðŸŽ¨ Point cloud & mesh visualization</div>
                    <div>ðŸ“ Real-time 3D manipulation</div>
                    <div>ðŸ“Š GIS coordinate integration</div>
                    <div>âš¡ Hardware-accelerated rendering</div>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Viewer Status Bar */}
        <div className="bg-white border-t border-gray-200 px-4 py-3">
          <div className="flex items-center justify-between text-sm">
            <div className="flex items-center space-x-6 text-gray-600">
              {results.pointCloud && (
                <div className="flex items-center">
                  <div className="w-2 h-2 bg-green-500 rounded-full mr-2"></div>
                  <strong>{results.pointCloud.points.toLocaleString()}</strong> 
                  <span className="ml-1">points loaded</span>
                </div>
              )}
              <div className="flex items-center">
                <Map className="w-4 h-4 mr-1 text-purple-500" />
                <span>Coord: {settings.coordinateSystem}</span>
              </div>
            </div>
            
            <div className="flex items-center space-x-4 text-xs">
              <div className="flex items-center">
                <div className="w-3 h-3 bg-red-500 rounded mr-1 shadow-sm"></div>
                <span className="font-medium">X-Axis</span>
              </div>
              <div className="flex items-center">
                <div className="w-3 h-3 bg-green-500 rounded mr-1 shadow-sm"></div>
                <span className="font-medium">Y-Axis</span>
              </div>
              <div className="flex items-center">
                <div className="w-3 h-3 bg-blue-500 rounded mr-1 shadow-sm"></div>
                <span className="font-medium">Z-Axis</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Settings Modal */}
      {showSettingsModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center">
          <div className="absolute inset-0 bg-black/50" onClick={() => setShowSettingsModal(false)} />
          <div className="relative bg-white rounded-2xl shadow-2xl w-full max-w-xl mx-4 p-6 max-h-[80vh] overflow-y-auto">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold">Settings</h3>
              <button onClick={() => setShowSettingsModal(false)} className="text-gray-500 hover:text-gray-700">âœ•</button>
            </div>
            <div className="space-y-5">
              <div>
                <label className="block text-xs font-medium text-gray-600 mb-2">Output Format</label>
                <div className="grid grid-cols-2 gap-2">
                  {['las', 'ply', 'xyz', 'laz', 'mesh_ply'].map(format => (
                    <button
                      key={format}
                      onClick={() => setSettings(prev => ({ ...prev, outputFormat: format }))}
                      className={`py-2 px-3 text-xs rounded-lg border transition-all ${
                        settings.outputFormat === format ? 'border-blue-500 bg-blue-500 text-white' : 'border-gray-300 text-gray-600 hover:border-gray-400'
                      }`}
                    >
                      {format.toUpperCase()}
                    </button>
                  ))}
                </div>
              </div>

              <div>
                <label className="block text-xs font-medium text-gray-600 mb-2">Point Density</label>
                <div className="flex space-x-1">
                  {['low', 'medium', 'high'].map(density => (
                    <button
                      key={density}
                      onClick={() => setSettings(prev => ({ ...prev, pointDensity: density }))}
                      className={`flex-1 py-1 px-2 text-xs rounded transition-all ${
                        settings.pointDensity === density ? 'bg-green-500 text-white' : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                      }`}
                    >
                      {density}
                    </button>
                  ))}
                </div>
              </div>

              <div>
                <label className="block text-xs font-medium text-gray-600 mb-2">Coordinate System</label>
                <select
                  value={settings.coordinateSystem}
                  onChange={(e) => setSettings(prev => ({ ...prev, coordinateSystem: e.target.value }))}
                  className="w-full text-sm border border-gray-300 rounded-lg px-3 py-2 focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                >
                  <option value="WGS84">WGS84 (EPSG:4326)</option>
                  <option value="UTM">UTM (Auto-detect zone)</option>
                  <option value="WebMercator">Web Mercator (EPSG:3857)</option>
                </select>
              </div>

              <div className="grid grid-cols-2 gap-3">
                <label className="flex items-center gap-2 text-xs text-gray-700">
                  <input type="checkbox" checked={settings.invertDepth} onChange={(e) => setSettings(prev => ({ ...prev, invertDepth: e.target.checked }))} />
                  Invert Depth
                </label>
                <label className="flex items-center gap-2 text-xs text-gray-700">
                  <input type="checkbox" checked={settings.smoothDepth} onChange={(e) => setSettings(prev => ({ ...prev, smoothDepth: e.target.checked }))} />
                  Smooth Depth
                </label>
              </div>

              <div>
                <label className="block text-xs font-medium text-gray-600 mb-1">Depth Scale: {settings.depthScale}</label>
                <input type="range" min="5" max="50" step="1" value={settings.depthScale} onChange={(e) => setSettings(prev => ({ ...prev, depthScale: Number(e.target.value) }))} className="w-full" />
              </div>

              <div>
                <label className="block text-xs font-medium text-gray-600 mb-1">Camera FOV: {settings.fov}Â°</label>
                <input type="range" min="30" max="90" step="1" value={settings.fov} onChange={(e) => setSettings(prev => ({ ...prev, fov: Number(e.target.value) }))} className="w-full" />
              </div>
            </div>
            <div className="mt-6 flex justify-end">
              <button onClick={() => setShowSettingsModal(false)} className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700">Close</button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default App;






