import React, { useState, useRef, useCallback, useEffect } from 'react';
import { Upload, Download, Settings, Play, Eye, Map, Loader2, CheckCircle2, RotateCcw, Grid, Box, Layers, Image, Zap, Brain, Cpu, FileText, Camera, Expand, BarChart3, Activity, ZoomIn, ZoomOut, Move, Info } from 'lucide-react';
import * as THREE from 'three';

const App = () => {
  const [uploadedImage, setUploadedImage] = useState(null);
  const [processing, setProcessing] = useState(false);
  const [currentStep, setCurrentStep] = useState(0);
  const [results, setResults] = useState({
    depthMap: null,
    pointCloud: null,
    mesh: null,
    gisData: null,
    jobId: null,
    preview: null,
    meshPreview: null
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
  const [showDepthModal, setShowDepthModal] = useState(false);
  
  // Depth map viewer state
  const [depthZoom, setDepthZoom] = useState(1);
  const [depthPan, setDepthPan] = useState({ x: 0, y: 0 });
  const [isPanning, setIsPanning] = useState(false);
  const [lastPanPoint, setLastPanPoint] = useState({ x: 0, y: 0 });
  const [hoverDepth, setHoverDepth] = useState(null);
  
  // Debug state for 3D viewer
  const [debugInfo, setDebugInfo] = useState({
    pointsLoaded: 0,
    sceneObjects: 0,
    cameraPosition: 'Not set'
  });
  
  const fileInputRef = useRef(null);
  const mountRef = useRef(null);
  const sceneRef = useRef(null);
  const rendererRef = useRef(null);
  const cameraRef = useRef(null);
  const pointCloudRef = useRef(null);
  const meshRef = useRef(null);
  const controlsRef = useRef({ mouseDown: false, cameraDistance: 8 });
  const depthCanvasRef = useRef(null);
  const API_BASE = 'http://localhost:8000';

  const steps = [
    { 
      name: 'Upload Image', 
      icon: Upload, 
      status: 'pending',
      description: 'Select your input image'
    },
    { 
      name: 'AI Analysis', 
      icon: Brain, 
      status: 'pending',
      description: 'Depth estimation with AI'
    },
    { 
      name: 'Generate 3D', 
      icon: Box, 
      status: 'pending',
      description: 'Create point cloud'
    },
    { 
      name: 'Export Ready', 
      icon: Download, 
      status: 'pending',
      description: 'GIS-ready output'
    }
  ];

  const models = [
    { 
      id: 'depth-anything-v2', 
      name: 'Depth Anything V2', 
      description: 'Superior depth estimation → point cloud',
      license: 'Apache-2.0',
      speed: '2-3s',
      recommended: true,
      quality: 'High',
      icon: Brain
    },
    { 
      id: 'triposr', 
      name: 'TripoSR', 
      description: 'Ultra-fast mesh generation',
      license: 'MIT',
      speed: '1-2s',
      quality: 'Medium',
      icon: Zap
    },
    { 
      id: 'instantmesh', 
      name: 'InstantMesh', 
      description: 'High-quality 3D assets',
      license: 'Custom',
      speed: '~10s',
      quality: 'Very High',
      icon: Cpu
    }
  ];

  // Initialize Three.js scene
  useEffect(() => {
    if (!mountRef.current) return;

    console.log('🎮 Initializing Three.js scene...');

    // Scene setup
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0a0b0f);
    sceneRef.current = scene;

    // Camera setup
    const camera = new THREE.PerspectiveCamera(75, 1, 0.1, 1000);
    camera.position.set(10, 8, 10);
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
    renderer.setClearColor(0x0a0b0f, 1);
    mountRef.current.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    // Enhanced lighting setup
    const ambientLight = new THREE.AmbientLight(0x404040, 0.6);
    scene.add(ambientLight);

    const directionalLight = new THREE.DirectionalLight(0xffffff, 1.0);
    directionalLight.position.set(10, 10, 5);
    directionalLight.castShadow = true;
    directionalLight.shadow.mapSize.width = 2048;
    directionalLight.shadow.mapSize.height = 2048;
    scene.add(directionalLight);

    const rimLight = new THREE.DirectionalLight(0x0099ff, 0.4);
    rimLight.position.set(-10, -5, -5);
    scene.add(rimLight);

    const fillLight = new THREE.DirectionalLight(0xff6600, 0.3);
    fillLight.position.set(5, -5, 10);
    scene.add(fillLight);

    // Grid and axes
    if (showGrid) {
      const gridHelper = new THREE.GridHelper(20, 20, 0x444466, 0x222233);
      gridHelper.material.transparent = true;
      gridHelper.material.opacity = 0.6;
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
      
      // Update debug info
      setDebugInfo(prev => ({
        ...prev,
        cameraPosition: `${camera.position.x.toFixed(1)}, ${camera.position.y.toFixed(1)}, ${camera.position.z.toFixed(1)}`
      }));
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
      
      // Update debug info
      setDebugInfo(prev => ({
        ...prev,
        cameraPosition: `${camera.position.x.toFixed(1)}, ${camera.position.y.toFixed(1)}, ${camera.position.z.toFixed(1)}`
      }));
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

    // Update debug info
    setDebugInfo(prev => ({
      ...prev,
      sceneObjects: scene.children.length,
      cameraPosition: `${camera.position.x.toFixed(1)}, ${camera.position.y.toFixed(1)}, ${camera.position.z.toFixed(1)}`
    }));

    console.log('✅ Three.js scene initialized');

    return () => {
      if (animationId) cancelAnimationFrame(animationId);
      if (mountRef.current && renderer.domElement && mountRef.current.contains(renderer.domElement)) {
        mountRef.current.removeChild(renderer.domElement);
      }
      window.removeEventListener('resize', handleResize);
      renderer.dispose();
    };
  }, [showGrid, showAxes, cameraAutoRotate]);

  // Handle 3D data loading and display
  useEffect(() => {
    console.log('🔄 3D data effect triggered', { 
      previewPoints: results?.preview?.points?.length, 
      meshVertices: results?.meshPreview?.vertices?.length,
      viewerMode 
    });

    if (!sceneRef.current) {
      console.log('❌ Scene not ready');
      return;
    }

    // Clear any loading state
    setViewer3DLoading(false);

    // If backend preview is present, render it directly for point clouds
    if (viewerMode === 'pointcloud' && results?.preview?.points?.length) {
      console.log('📊 Loading point cloud from preview data');
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
      console.log('🔺 Loading mesh from preview data');
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
      console.log('🌐 Fetching 3D data from backend');
      load3DDataFromBackend(results.jobId);
      return;
    }
    
    // Fallback demo - only if no real data
    if (!results.pointCloud && !results.mesh && !results.preview) {
      console.log('🎪 Generating demo 3D content');
      generateDemo3DContent();
    }
  }, [results.preview, results.pointCloud, results.mesh, results.meshPreview, results.jobId, pointSize, viewerMode, wireframe]);

  const load3DDataFromBackend = async (jobId) => {
    setViewer3DLoading(true);
    try {
      console.log('🔍 Fetching job status:', jobId);
      const res = await fetch(`${API_BASE}/status/${jobId}`);
      if (!res.ok) throw new Error('Status fetch failed');
      const s = await res.json();
      
      console.log('📥 Backend response:', s);
      
      if (s?.results?.preview && viewerMode === 'pointcloud') {
        const pts = s.results.preview.points || [];
        const cols = s.results.preview.colors || [];
        console.log(`📊 Processing ${pts.length} points from backend`);
        
        if (pts.length > 0) {
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
        } else {
          console.log('⚠️ No points received from backend');
          generateDemo3DContent();
        }
      } else if (viewerMode === 'mesh') {
        console.log('🔺 Mesh mode but no mesh data, generating demo');
        generateDemo3DContent();
      }
    } catch (error) {
      console.error('❌ Error loading 3D data:', error);
      generateDemo3DContent();
    } finally {
      setViewer3DLoading(false);
    }
  };

  const clearScene = () => {
    if (!sceneRef.current) return;
    
    console.log('🧹 Clearing 3D scene');
    
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
    
    // Update debug info
    setDebugInfo(prev => ({
      ...prev,
      pointsLoaded: 0,
      sceneObjects: sceneRef.current.children.length
    }));
  };

  const displayPointCloudData = (data) => {
    if (!sceneRef.current) {
      console.log('❌ Scene not available for point cloud display');
      return;
    }
    
    console.log('🎨 Displaying point cloud data', { 
      pointsLength: data.points.length / 3,
      colorsLength: data.colors.length / 3 
    });
    
    // Always clear scene first to remove old content
    clearScene();

    try {
      const geometry = new THREE.BufferGeometry();
      const positions = new Float32Array(data.points);
      const colors = new Float32Array(data.colors);

      geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
      geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

      const material = new THREE.PointsMaterial({
        size: Math.max(0.8, pointSize / 8),
        vertexColors: true,
        sizeAttenuation: true,
        transparent: true,
        opacity: 0.9
      });

      const points = new THREE.Points(geometry, material);
      pointCloudRef.current = points;
      sceneRef.current.add(points);

      console.log('✅ Point cloud added to scene');

      // Improved automatic camera positioning
      try {
        geometry.computeBoundingBox();
        const box = geometry.boundingBox;
        if (box && !box.isEmpty()) {
          const center = box.getCenter(new THREE.Vector3());
          const size = box.getSize(new THREE.Vector3());
          const maxDim = Math.max(size.x, size.y, size.z);
          
          if (maxDim > 0) {
            const fov = cameraRef.current.fov * (Math.PI / 180);
            const cameraDistance = Math.abs(maxDim / (2 * Math.tan(fov / 2))) * 2.2; // Increased padding
            
            // Position camera at a good angle
            const offset = new THREE.Vector3(
              cameraDistance * 0.6,
              cameraDistance * 0.4,
              cameraDistance * 0.6
            );
            
            cameraRef.current.position.copy(center).add(offset);
            cameraRef.current.lookAt(center);
            controlsRef.current.cameraDistance = cameraDistance;
            
            console.log('📷 Auto-positioned camera for point cloud', { 
              center: center.toArray(), 
              size: maxDim, 
              cameraDistance,
              cameraPos: cameraRef.current.position.toArray()
            });
          }
        } else {
          console.log('⚠️ Empty bounding box, using fallback positioning');
          // Fallback positioning
          cameraRef.current.position.set(15, 10, 15);
          cameraRef.current.lookAt(0, 0, 0);
          controlsRef.current.cameraDistance = 20;
        }
      } catch (err) {
        console.log('⚠️ Could not auto-position camera:', err);
        // Fallback positioning
        cameraRef.current.position.set(15, 10, 15);
        cameraRef.current.lookAt(0, 0, 0);
        controlsRef.current.cameraDistance = 20;
      }

      // Update debug info
      setDebugInfo(prev => ({
        ...prev,
        pointsLoaded: positions.length / 3,
        sceneObjects: sceneRef.current.children.length,
        cameraPosition: `${cameraRef.current.position.x.toFixed(1)}, ${cameraRef.current.position.y.toFixed(1)}, ${cameraRef.current.position.z.toFixed(1)}`
      }));

    } catch (error) {
      console.error('❌ Error displaying point cloud:', error);
    }
  };

  const displayMeshData = (data) => {
    if (!sceneRef.current) return;
    
    console.log('🔺 Displaying mesh data');
    
    // Always clear scene first to remove old content
    clearScene();

    try {
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
        opacity: wireframe ? 1.0 : 0.9,
        side: THREE.DoubleSide,
        shininess: 30
      });

      const mesh = new THREE.Mesh(geometry, material);
      mesh.castShadow = true;
      mesh.receiveShadow = true;
      meshRef.current = mesh;
      sceneRef.current.add(mesh);

      console.log('✅ Mesh added to scene');

      // Improved automatic camera positioning for mesh
      try {
        geometry.computeBoundingBox();
        const box = geometry.boundingBox;
        if (box && !box.isEmpty()) {
          const center = box.getCenter(new THREE.Vector3());
          const size = box.getSize(new THREE.Vector3());
          const maxDim = Math.max(size.x, size.y, size.z);
          
          if (maxDim > 0) {
            const fov = cameraRef.current.fov * (Math.PI / 180);
            const cameraDistance = Math.abs(maxDim / (2 * Math.tan(fov / 2))) * 2.5; // More padding for mesh
            
            // Position camera at a good angle for mesh viewing
            const offset = new THREE.Vector3(
              cameraDistance * 0.7,
              cameraDistance * 0.5,
              cameraDistance * 0.7
            );
            
            cameraRef.current.position.copy(center).add(offset);
            cameraRef.current.lookAt(center);
            controlsRef.current.cameraDistance = cameraDistance;
            
            console.log('📷 Auto-positioned camera for mesh', { 
              center: center.toArray(), 
              size: maxDim, 
              cameraDistance,
              cameraPos: cameraRef.current.position.toArray()
            });
          }
        } else {
          console.log('⚠️ Empty bounding box for mesh, using fallback positioning');
          cameraRef.current.position.set(15, 10, 15);
          cameraRef.current.lookAt(0, 0, 0);
          controlsRef.current.cameraDistance = 20;
        }
      } catch (err) {
        console.log('⚠️ Could not auto-position camera for mesh:', err);
        cameraRef.current.position.set(15, 10, 15);
        cameraRef.current.lookAt(0, 0, 0);
        controlsRef.current.cameraDistance = 20;
      }

      // Update debug info
      setDebugInfo(prev => ({
        ...prev,
        pointsLoaded: positions.length / 3,
        sceneObjects: sceneRef.current.children.length,
        cameraPosition: `${cameraRef.current.position.x.toFixed(1)}, ${cameraRef.current.position.y.toFixed(1)}, ${cameraRef.current.position.z.toFixed(1)}`
      }));

    } catch (error) {
      console.error('❌ Error displaying mesh:', error);
    }
  };

  const generateDemo3DContent = () => {
    if (!sceneRef.current) return;
    
    console.log('🎪 Generating demo 3D content for', viewerMode);
    clearScene();

    if (viewerMode === 'pointcloud') {
      const pointCount = 8000;
      const geometry = new THREE.BufferGeometry();
      const positions = new Float32Array(pointCount * 3);
      const colors = new Float32Array(pointCount * 3);

      for (let i = 0; i < pointCount; i++) {
        const i3 = i * 3;
        const x = (Math.random() - 0.5) * 12;
        const z = (Math.random() - 0.5) * 12;
        const y = Math.sin(x * 0.3) * Math.cos(z * 0.3) * 3 + Math.random() * 0.8;
        
        positions[i3] = x;
        positions[i3 + 1] = y;
        positions[i3 + 2] = z;

        const heightNorm = (y + 3) / 6;
        colors[i3] = Math.max(0.2, heightNorm - 0.2);
        colors[i3 + 1] = Math.max(0.3, 1 - Math.abs(heightNorm - 0.5) * 2);
        colors[i3 + 2] = Math.max(0.4, 0.8 - heightNorm);
      }

      geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
      geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

      const material = new THREE.PointsMaterial({
        size: pointSize / 8,
        vertexColors: true,
        sizeAttenuation: true,
        transparent: true,
        opacity: 0.8
      });

      const points = new THREE.Points(geometry, material);
      pointCloudRef.current = points;
      sceneRef.current.add(points);
      
      console.log('✅ Demo point cloud created');
      
    } else if (viewerMode === 'mesh') {
      const geometry = new THREE.SphereGeometry(4, 32, 32);
      
      const positions = geometry.attributes.position;
      for (let i = 0; i < positions.count; i++) {
        const vertex = new THREE.Vector3();
        vertex.fromBufferAttribute(positions, i);
        const noise = (Math.random() - 0.5) * 0.4;
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
      
      console.log('✅ Demo mesh created');
    }

    // Update debug info
    setDebugInfo(prev => ({
      ...prev,
      pointsLoaded: viewerMode === 'pointcloud' ? 8000 : 1032,
      sceneObjects: sceneRef.current.children.length
    }));
  };

  const handleImageUpload = useCallback((event) => {
    const file = event.target.files[0];
    if (file) {
      if (file.size > 50 * 1024 * 1024) {
        alert('File size must be less than 50MB');
        return;
      }
      
      // Clear previous results when uploading new image
      setResults({
        depthMap: null,
        pointCloud: null,
        mesh: null,
        gisData: null,
        jobId: null,
        preview: null,
        meshPreview: null
      });
      
      // Clear 3D viewer
      if (sceneRef.current) {
        clearScene();
      }
      
      // Reset steps
      steps.forEach((step, index) => {
        step.status = index === 0 ? 'completed' : 'pending';
      });
      setCurrentStep(0);
      
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
      form.append('invert_depth', String(settings.invertDepth));
      form.append('depth_scale', String(settings.depthScale));
      form.append('smooth_depth', String(settings.smoothDepth));
      form.append('fov', String(settings.fov));

      console.log('🚀 Starting image processing...');

      const startRes = await fetch(`${API_BASE}/process`, {
        method: 'POST',
        body: form
      });
      
      if (!startRes.ok) throw new Error('Failed to start processing');
      const startData = await startRes.json();
      const id = startData?.job_id;
      setResults(prev => ({ ...prev, jobId: id }));

      console.log('📋 Job started:', id);

      const poll = async () => {
        const res = await fetch(`${API_BASE}/status/${id}`);
        if (!res.ok) {
          setProcessing(false);
          return;
        }
        const s = await res.json();
        console.log('📊 Job progress:', s.progress, s.message);
        
        if (s.progress >= 1 && s.progress < 40) updateStepStatus(1, 'processing');
        if (s.progress >= 40 && s.progress < 80) updateStepStatus(2, 'processing');
        if (s.progress >= 80 && s.progress < 100) updateStepStatus(3, 'processing');
        if (s.status === 'completed') {
          updateStepStatus(1, 'completed');
          updateStepStatus(2, 'completed');
          updateStepStatus(3, 'completed');
          
          console.log('✅ Processing completed:', s.results);
          
          setResults(prev => ({
            ...prev,
            pointCloud: s.results?.pointCloud || null,
            gisData: s.results?.gisData || null,
            depthMap: s.results?.depthMap || null,
            preview: s.results?.preview || null,
            meshPreview: s.results?.meshPreview || null,
            downloadUrl: s.results?.downloadUrl ? `${API_BASE}${s.results.downloadUrl}` : null,
          }));
          setProcessing(false);
          return;
        }
        if (s.status === 'error') {
          console.error('❌ Processing error:', s.message);
          setProcessing(false);
          return;
        }
        setTimeout(poll, 1500);
      };
      poll();
    } catch (e) {
      console.error('❌ Failed to start processing', e);
      setProcessing(false);
    }
  };

  const resetCamera = () => {
    if (cameraRef.current) {
      cameraRef.current.position.set(10, 8, 10);
      cameraRef.current.lookAt(0, 0, 0);
      controlsRef.current.cameraDistance = 12;
      
      setDebugInfo(prev => ({
        ...prev,
        cameraPosition: `${cameraRef.current.position.x.toFixed(1)}, ${cameraRef.current.position.y.toFixed(1)}, ${cameraRef.current.position.z.toFixed(1)}`
      }));
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

  // Depth map interactive controls
  const handleDepthWheel = (e) => {
    e.preventDefault();
    e.stopPropagation();
    const delta = e.deltaY > 0 ? 0.9 : 1.1;
    setDepthZoom(prev => Math.min(Math.max(prev * delta, 0.5), 10));
  };

  const handleDepthMouseDown = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsPanning(true);
    setLastPanPoint({ x: e.clientX, y: e.clientY });
    document.body.style.userSelect = 'none'; // Prevent text selection
  };

  const handleDepthMouseMove = (e) => {
    e.preventDefault();
    e.stopPropagation();
    
    if (isPanning) {
      const deltaX = e.clientX - lastPanPoint.x;
      const deltaY = e.clientY - lastPanPoint.y;
      setDepthPan(prev => ({
        x: prev.x + deltaX / depthZoom,
        y: prev.y + deltaY / depthZoom
      }));
      setLastPanPoint({ x: e.clientX, y: e.clientY });
    }

    // Show depth value on hover (simplified)
    const rect = e.currentTarget.getBoundingClientRect();
    const x = ((e.clientX - rect.left) / rect.width * 100).toFixed(1);
    const y = ((e.clientY - rect.top) / rect.height * 100).toFixed(1);
    setHoverDepth({ x, y });
  };

  const handleDepthMouseUp = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsPanning(false);
    document.body.style.userSelect = ''; // Restore text selection
  };

  const resetDepthView = () => {
    setDepthZoom(1);
    setDepthPan({ x: 0, y: 0 });
  };

  // 3D Viewer zoom to extent functionality
  const zoomToExtent = () => {
    if (!cameraRef.current || !sceneRef.current) return;
    
    console.log('🎯 Zooming to extent...');
    
    // Find all visible geometry in the scene
    const box = new THREE.Box3();
    let hasGeometry = false;
    
    sceneRef.current.traverse((object) => {
      if (object.geometry && (object instanceof THREE.Points || object instanceof THREE.Mesh)) {
        if (!object.geometry.boundingBox) {
          object.geometry.computeBoundingBox();
        }
        if (object.geometry.boundingBox) {
          box.expandByObject(object);
          hasGeometry = true;
        }
      }
    });
    
    if (!hasGeometry) {
      console.log('⚠️ No geometry found for zoom to extent');
      return;
    }
    
    const center = box.getCenter(new THREE.Vector3());
    const size = box.getSize(new THREE.Vector3());
    const maxDim = Math.max(size.x, size.y, size.z);
    
    if (maxDim === 0) {
      console.log('⚠️ Geometry has no size');
      return;
    }
    
    // Calculate optimal camera distance
    const camera = cameraRef.current;
    const fov = camera.fov * (Math.PI / 180);
    const distance = Math.abs(maxDim / (2 * Math.tan(fov / 2))) * 2.5; // Add some padding
    
    // Position camera to view the entire object
    const direction = new THREE.Vector3().subVectors(camera.position, center).normalize();
    if (direction.length() === 0) {
      direction.set(1, 1, 1).normalize();
    }
    
    camera.position.copy(center).add(direction.multiplyScalar(distance));
    camera.lookAt(center);
    
    // Update controls
    controlsRef.current.cameraDistance = distance;
    
    console.log('✅ Zoomed to extent:', { center, size: maxDim, distance });
    
    // Update debug info
    setDebugInfo(prev => ({
      ...prev,
      cameraPosition: `${camera.position.x.toFixed(1)}, ${camera.position.y.toFixed(1)}, ${camera.position.z.toFixed(1)}`
    }));
  };

  // Demo function for testing without backend
  const generateDemoResults = () => {
    console.log('🎪 Generating demo results...');
    
    // Generate demo depth map (gray gradient)
    const canvas = document.createElement('canvas');
    canvas.width = 400;
    canvas.height = 300;
    const ctx = canvas.getContext('2d');
    const gradient = ctx.createLinearGradient(0, 0, 400, 300);
    gradient.addColorStop(0, '#000033');
    gradient.addColorStop(0.5, '#0066cc');
    gradient.addColorStop(1, '#ffffff');
    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, 400, 300);
    const demoDepthMap = canvas.toDataURL();

    // Generate demo point cloud data
    const pointCount = 5000;
    const demoPoints = [];
    const demoColors = [];
    
    for (let i = 0; i < pointCount; i++) {
      const x = (Math.random() - 0.5) * 10;
      const z = (Math.random() - 0.5) * 10;
      const y = Math.sin(x * 0.3) * Math.cos(z * 0.3) * 2 + Math.random() * 0.5;
      
      demoPoints.push([x, y, z]);
      
      const heightNorm = (y + 2) / 4;
      demoColors.push([
        Math.floor(255 * Math.max(0, heightNorm - 0.3)),
        Math.floor(255 * Math.max(0, 1 - Math.abs(heightNorm - 0.5) * 2)),
        Math.floor(255 * Math.max(0, 0.7 - heightNorm))
      ]);
    }

    setResults({
      depthMap: demoDepthMap,
      pointCloud: {
        points: pointCount,
        format: 'LAS'
      },
      preview: {
        points: demoPoints,
        colors: demoColors
      },
      gisData: {
        coordinateSystem: 'WGS84',
        bounds: {
          minX: -5, maxX: 5,
          minY: -2, maxY: 2,
          minZ: -5, maxZ: 5
        },
        pointCount: pointCount
      },
      downloadUrl: '#demo'
    });

    // Update all steps to completed
    updateStepStatus(0, 'completed');
    updateStepStatus(1, 'completed');
    updateStepStatus(2, 'completed');
    updateStepStatus(3, 'completed');
    
    console.log('✅ Demo results generated');
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-blue-900 to-slate-900 text-white">
      {/* Header */}
      <div className="bg-slate-900/80 backdrop-blur-sm border-b border-slate-700/50">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <div className="w-12 h-12 bg-gradient-to-br from-blue-500 to-purple-600 rounded-xl flex items-center justify-center">
                <Box className="w-6 h-6 text-white" />
              </div>
              <div>
                <h1 className="text-2xl font-bold bg-gradient-to-r from-white to-blue-200 bg-clip-text text-transparent">
                  AI Point Cloud Generator
                </h1>
                <p className="text-slate-400 text-sm">Professional Image → 3D → GIS Pipeline</p>
              </div>
            </div>
            <div className="flex items-center space-x-4">
              <button
                onClick={generateDemoResults}
                className="px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition-colors text-sm"
              >
                Demo Mode
              </button>
              <div className="text-right">
                <div className="text-sm text-slate-300">Powered by</div>
                <div className="text-xs text-slate-400">Depth Anything V2 • Three.js • FastAPI</div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto p-6 grid grid-cols-12 gap-6">
        
        {/* Left Panel - Upload & Settings */}
        <div className="col-span-4 space-y-6">
          
          {/* Progress Steps */}
          <div className="bg-slate-800/50 rounded-2xl p-6 border border-slate-700/50 backdrop-blur-sm">
            <h3 className="text-lg font-semibold mb-4 flex items-center">
              <Activity className="w-5 h-5 mr-2 text-blue-400" />
              Processing Pipeline
            </h3>
            <div className="space-y-3">
              {steps.map((step, index) => {
                const Icon = step.icon;
                const isActive = index === currentStep && processing;
                const isCompleted = step.status === 'completed';
                const isProcessing = step.status === 'processing';
                
                return (
                  <div key={step.name} className={`relative flex items-center space-x-4 p-4 rounded-xl transition-all duration-300 ${
                    isActive ? 'bg-blue-500/20 border border-blue-500/30' : 
                    isCompleted ? 'bg-green-500/20 border border-green-500/30' :
                    'bg-slate-700/30 border border-slate-600/30'
                  }`}>
                    <div className={`flex items-center justify-center w-10 h-10 rounded-full transition-all duration-300 ${
                      isCompleted 
                        ? 'bg-green-500 text-white shadow-lg shadow-green-500/30' 
                        : isProcessing 
                          ? 'bg-blue-500 text-white shadow-lg shadow-blue-500/30' 
                          : 'bg-slate-600 text-slate-300'
                    }`}>
                      {isProcessing ? (
                        <Loader2 className="w-5 h-5 animate-spin" />
                      ) : isCompleted ? (
                        <CheckCircle2 className="w-5 h-5" />
                      ) : (
                        <Icon className="w-5 h-5" />
                      )}
                    </div>
                    <div className="flex-1">
                      <div className={`font-medium ${
                        isActive ? 'text-blue-200' : isCompleted ? 'text-green-200' : 'text-slate-300'
                      }`}>
                        {step.name}
                      </div>
                      <div className="text-xs text-slate-400">{step.description}</div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Image Upload */}
          <div className="bg-slate-800/50 rounded-2xl p-6 border border-slate-700/50 backdrop-blur-sm">
            <h3 className="text-lg font-semibold mb-4 flex items-center">
              <Upload className="w-5 h-5 mr-2 text-green-400" />
              Input Image
            </h3>
            
            {!uploadedImage ? (
              <div
                onClick={() => fileInputRef.current?.click()}
                className="group border-2 border-dashed border-slate-600 rounded-xl p-8 text-center cursor-pointer hover:border-blue-400 hover:bg-blue-400/5 transition-all duration-300"
              >
                <Upload className="w-12 h-12 text-slate-400 mx-auto mb-4 group-hover:text-blue-400 transition-colors" />
                <p className="text-slate-300 mb-2 font-medium">Drop your image here</p>
                <p className="text-sm text-slate-400">PNG, JPG, WEBP • Max 50MB</p>
                <div className="mt-4">
                  <span className="inline-flex items-center px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors">
                    <Camera className="w-4 h-4 mr-2" />
                    Choose Image
                  </span>
                </div>
              </div>
            ) : (
              <div className="space-y-4">
                <div className="relative group">
                  <img
                    src={uploadedImage.url}
                    alt="Uploaded"
                    className="w-full h-48 object-cover rounded-xl"
                  />
                  <div className="absolute inset-0 bg-black/40 opacity-0 group-hover:opacity-100 transition-opacity rounded-xl flex items-center justify-center">
                    <button
                      onClick={() => fileInputRef.current?.click()}
                      className="px-4 py-2 bg-white/20 backdrop-blur-sm text-white rounded-lg hover:bg-white/30 transition-colors"
                    >
                      Change Image
                    </button>
                  </div>
                  <div className="absolute top-3 right-3 bg-black/60 text-white px-3 py-1 rounded-lg text-sm backdrop-blur-sm">
                    {uploadedImage.size}
                  </div>
                </div>
                <div className="bg-slate-700/50 rounded-lg p-3">
                  <div className="text-sm font-medium text-slate-200 truncate">{uploadedImage.name}</div>
                  <div className="text-xs text-slate-400">Ready for processing</div>
                </div>
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

          {/* Model Selection */}
          <div className="bg-slate-800/50 rounded-2xl p-6 border border-slate-700/50 backdrop-blur-sm">
            <h3 className="text-lg font-semibold mb-4 flex items-center">
              <Brain className="w-5 h-5 mr-2 text-purple-400" />
              AI Model
            </h3>
            <div className="space-y-3">
              {models.map((model) => {
                const Icon = model.icon;
                const isSelected = settings.model === model.id;
                return (
                  <div
                    key={model.id}
                    onClick={() => setSettings(prev => ({ ...prev, model: model.id }))}
                    className={`p-4 rounded-xl cursor-pointer transition-all duration-200 border ${
                      isSelected 
                        ? 'bg-blue-500/20 border-blue-500/50 shadow-lg shadow-blue-500/20' 
                        : 'bg-slate-700/30 border-slate-600/30 hover:bg-slate-700/50'
                    }`}
                  >
                    <div className="flex items-start space-x-3">
                      <Icon className={`w-6 h-6 mt-1 ${isSelected ? 'text-blue-400' : 'text-slate-400'}`} />
                      <div className="flex-1">
                        <div className="flex items-center space-x-2">
                          <span className={`font-medium ${isSelected ? 'text-blue-200' : 'text-slate-200'}`}>
                            {model.name}
                          </span>
                          {model.recommended && (
                            <span className="px-2 py-1 bg-green-500/20 text-green-300 text-xs rounded-full">
                              Recommended
                            </span>
                          )}
                        </div>
                        <p className="text-sm text-slate-400 mt-1">{model.description}</p>
                        <div className="flex items-center space-x-4 mt-2 text-xs">
                          <span className="text-slate-500">Speed: {model.speed}</span>
                          <span className="text-slate-500">Quality: {model.quality}</span>
                        </div>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Generate Button */}
          {uploadedImage && (
            <button
              onClick={processImage}
              disabled={processing}
              className="w-full bg-gradient-to-r from-blue-600 to-purple-600 text-white py-4 px-6 rounded-xl hover:from-blue-700 hover:to-purple-700 disabled:from-gray-600 disabled:to-gray-700 disabled:cursor-not-allowed transition-all duration-200 flex items-center justify-center gap-3 text-lg font-semibold shadow-lg shadow-blue-500/30"
            >
              {processing ? (
                <>
                  <Loader2 className="w-6 h-6 animate-spin" />
                  Processing with AI...
                </>
              ) : (
                <>
                  <Zap className="w-6 h-6" />
                  Generate 3D Point Cloud
                </>
              )}
            </button>
          )}
        </div>

        {/* Middle Panel - Image/Depth View */}
        <div className="col-span-4 space-y-6">
          
          {/* Image Display */}
          {uploadedImage && (
            <div className="bg-slate-800/50 rounded-2xl p-6 border border-slate-700/50 backdrop-blur-sm">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-semibold flex items-center">
                  <Image className="w-5 h-5 mr-2 text-blue-400" />
                  Original Image
                </h3>
                <button
                  onClick={() => setShowDepthModal(true)}
                  className="p-2 bg-slate-700 rounded-lg hover:bg-slate-600 transition-colors"
                  title="View Full Size"
                >
                  <Expand className="w-4 h-4" />
                </button>
              </div>
              <img
                src={uploadedImage.url}
                alt="Original"
                className="w-full h-64 object-cover rounded-xl"
              />
            </div>
          )}

          {/* Interactive Depth Map Display */}
          {results.depthMap && (
            <div className="bg-slate-800/50 rounded-2xl p-6 border border-slate-700/50 backdrop-blur-sm">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-semibold flex items-center">
                  <Eye className="w-5 h-5 mr-2 text-purple-400" />
                  Interactive Depth Map
                </h3>
                <div className="flex items-center space-x-2">
                  <button
                    onClick={() => setDepthZoom(prev => Math.min(prev * 1.2, 10))}
                    className="p-2 bg-slate-700 rounded-lg hover:bg-slate-600 transition-colors"
                    title="Zoom In"
                  >
                    <ZoomIn className="w-4 h-4" />
                  </button>
                  <button
                    onClick={() => setDepthZoom(prev => Math.max(prev * 0.8, 0.5))}
                    className="p-2 bg-slate-700 rounded-lg hover:bg-slate-600 transition-colors"
                    title="Zoom Out"
                  >
                    <ZoomOut className="w-4 h-4" />
                  </button>
                  <button
                    onClick={resetDepthView}
                    className="p-2 bg-slate-700 rounded-lg hover:bg-slate-600 transition-colors"
                    title="Reset View"
                  >
                    <RotateCcw className="w-4 h-4" />
                  </button>
                  <button
                    onClick={() => setShowDepthModal(true)}
                    className="p-2 bg-slate-700 rounded-lg hover:bg-slate-600 transition-colors"
                    title="Full Screen"
                  >
                    <Expand className="w-4 h-4" />
                  </button>
                </div>
              </div>
              
              <div 
                className="relative h-64 overflow-hidden rounded-xl border border-slate-600"
                style={{ touchAction: 'none' }}
              >
                <div
                  className="w-full h-full cursor-move select-none"
                  style={{
                    transform: `translate(${depthPan.x}px, ${depthPan.y}px) scale(${depthZoom})`,
                    transformOrigin: 'center center'
                  }}
                  onWheel={handleDepthWheel}
                  onMouseDown={handleDepthMouseDown}
                  onMouseMove={handleDepthMouseMove}
                  onMouseUp={handleDepthMouseUp}
                  onMouseLeave={handleDepthMouseUp}
                  onContextMenu={(e) => e.preventDefault()}
                >
                  <img
                    src={results.depthMap}
                    alt="Depth Map"
                    className="w-full h-full object-contain pointer-events-none select-none"
                    draggable={false}
                  />
                </div>
                
                {/* Depth value overlay */}
                {hoverDepth && (
                  <div className="absolute top-2 left-2 bg-black/80 text-white px-2 py-1 rounded text-xs backdrop-blur-sm">
                    Position: {hoverDepth.x}%, {hoverDepth.y}%
                  </div>
                )}
                
                {/* Zoom indicator */}
                <div className="absolute bottom-2 right-2 bg-black/80 text-white px-2 py-1 rounded text-xs backdrop-blur-sm">
                  {(depthZoom * 100).toFixed(0)}%
                </div>
              </div>
              
              <div className="mt-3 flex items-center justify-between text-sm">
                <span className="text-slate-400">
                  Depth estimation: {settings.model}
                </span>
                <div className="flex items-center space-x-4 text-xs text-slate-500">
                  <span>🖱️ Drag: Pan</span>
                  <span>⚡ Wheel: Zoom</span>
                </div>
              </div>
            </div>
          )}

          {/* Quick Stats */}
          {results.pointCloud && (
            <div className="bg-slate-800/50 rounded-2xl p-6 border border-slate-700/50 backdrop-blur-sm">
              <h3 className="text-lg font-semibold mb-4 flex items-center">
                <BarChart3 className="w-5 h-5 mr-2 text-green-400" />
                Generation Stats
              </h3>
              <div className="grid grid-cols-2 gap-4">
                <div className="bg-slate-700/50 rounded-lg p-3 text-center">
                  <div className="text-2xl font-bold text-blue-400">
                    {results.pointCloud.points?.toLocaleString() || '0'}
                  </div>
                  <div className="text-sm text-slate-400">Points Generated</div>
                </div>
                <div className="bg-slate-700/50 rounded-lg p-3 text-center">
                  <div className="text-2xl font-bold text-purple-400">
                    {settings.outputFormat.toUpperCase()}
                  </div>
                  <div className="text-sm text-slate-400">Output Format</div>
                </div>
                <div className="bg-slate-700/50 rounded-lg p-3 text-center">
                  <div className="text-2xl font-bold text-green-400">
                    {settings.pointDensity}
                  </div>
                  <div className="text-sm text-slate-400">Density</div>
                </div>
                <div className="bg-slate-700/50 rounded-lg p-3 text-center">
                  <div className="text-2xl font-bold text-yellow-400">
                    {settings.coordinateSystem}
                  </div>
                  <div className="text-sm text-slate-400">Coord System</div>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Right Panel - 3D Viewer & Downloads */}
        <div className="col-span-4 space-y-6">
          
          {/* 3D Viewer */}
          <div className="bg-slate-800/50 rounded-2xl border border-slate-700/50 backdrop-blur-sm overflow-hidden">
            {/* Viewer Header */}
            <div className="p-4 border-b border-slate-700/50">
              <div className="flex items-center justify-between">
                <div className="flex items-center">
                  <Box className="w-5 h-5 text-blue-400 mr-2" />
                  <div>
                    <h3 className="text-lg font-semibold">3D Viewer</h3>
                    <p className="text-sm text-slate-400">
                      {viewerMode === 'pointcloud' ? 'Point Cloud' : 'Mesh'} Visualization
                    </p>
                  </div>
                </div>
                
                <div className="flex items-center space-x-2">
                  <div className="flex bg-slate-700/50 rounded-lg p-1">
                    <button
                      onClick={() => setViewerMode('pointcloud')}
                      className={`p-2 rounded transition-all ${
                        viewerMode === 'pointcloud' 
                          ? 'bg-blue-600 text-white shadow-lg' 
                          : 'text-slate-400 hover:text-white'
                      }`}
                      title="Point Cloud View"
                    >
                      <Grid className="w-4 h-4" />
                    </button>
                    <button
                      onClick={() => setViewerMode('mesh')}
                      className={`p-2 rounded transition-all ${
                        viewerMode === 'mesh' 
                          ? 'bg-blue-600 text-white shadow-lg' 
                          : 'text-slate-400 hover:text-white'
                      }`}
                      title="Mesh View"
                    >
                      <Box className="w-4 h-4" />
                    </button>
                  </div>
                  
                  <button
                    onClick={zoomToExtent}
                    className="p-2 bg-slate-700 rounded-lg hover:bg-slate-600 transition-colors"
                    title="Zoom to Extent - Fit All Content"
                  >
                    <Expand className="w-4 h-4" />
                  </button>
                  
                  <button
                    onClick={resetCamera}
                    className="p-2 bg-slate-700 rounded-lg hover:bg-slate-600 transition-colors"
                    title="Reset Camera Position"
                  >
                    <RotateCcw className="w-4 h-4" />
                  </button>
                  
                  <button
                    onClick={() => setShowSettingsModal(true)}
                    className="p-2 bg-slate-700 rounded-lg hover:bg-slate-600 transition-colors"
                    title="Viewer Settings"
                  >
                    <Settings className="w-4 h-4" />
                  </button>
                </div>
              </div>
            </div>

            {/* 3D Viewer Area */}
            <div className="relative h-96">
              {viewer3DLoading && (
                <div className="absolute inset-0 bg-slate-900/90 flex items-center justify-center z-20">
                  <div className="text-center">
                    <Loader2 className="w-12 h-12 mx-auto mb-3 animate-spin text-blue-400" />
                    <p className="text-lg font-semibold">Loading 3D Data</p>
                    <p className="text-sm text-slate-400">Processing {viewerMode}...</p>
                  </div>
                </div>
              )}
              
              <div className="relative w-full h-full">
                <div ref={mountRef} className="w-full h-full" />
                
                {/* Viewer Overlay Info */}
                <div className="absolute top-3 left-3 bg-black/60 text-white p-2 rounded-lg text-xs backdrop-blur-sm">
                  <div className="font-semibold text-blue-300">
                    {viewerMode === 'pointcloud' ? 'Point Cloud' : '3D Mesh'}
                  </div>
                  <div>Points: {debugInfo.pointsLoaded.toLocaleString()}</div>
                  <div>Objects: {debugInfo.sceneObjects}</div>
                </div>
                
                <div className="absolute bottom-3 left-3 bg-black/60 text-white p-2 rounded-lg text-xs backdrop-blur-sm">
                  <div>🖱️ Drag: Rotate</div>
                  <div>🎯 Wheel: Zoom</div>
                  <div>📍 Pos: {debugInfo.cameraPosition}</div>
                </div>
                
                <div className="absolute top-3 right-3 bg-black/60 text-white p-2 rounded-lg text-xs backdrop-blur-sm">
                  <div className="text-green-400 font-semibold">● WebGL Ready</div>
                  <div>Three.js Renderer</div>
                  <div>Mode: {viewerMode}</div>
                </div>
              </div>
            </div>
          </div>

          {/* Download Results */}
          {results.pointCloud && (
            <div className="bg-slate-800/50 rounded-2xl p-6 border border-slate-700/50 backdrop-blur-sm">
              <h3 className="text-lg font-semibold mb-4 flex items-center">
                <Download className="w-5 h-5 mr-2 text-green-400" />
                Download Results
              </h3>
              
              <div className="space-y-3">
                <div className="bg-gradient-to-r from-green-500/20 to-emerald-500/20 rounded-xl p-4 border border-green-500/30">
                  <div className="flex items-center justify-between mb-3">
                    <span className="font-medium text-green-200">Point Cloud</span>
                    <span className="text-xs text-green-300 bg-green-500/20 px-2 py-1 rounded">
                      {results.pointCloud.format}
                    </span>
                  </div>
                  <div className="text-sm text-green-300 mb-3">
                    <div>{results.pointCloud.points?.toLocaleString()} points</div>
                  </div>
                  <button
                    onClick={() => downloadFile('pointcloud')}
                    className="w-full bg-green-600 text-white py-2 px-4 rounded-lg hover:bg-green-700 transition-colors flex items-center justify-center font-medium"
                  >
                    <Download className="w-4 h-4 mr-2" />
                    Download {settings.outputFormat.toUpperCase()}
                  </button>
                </div>

                {results.gisData && (
                  <div className="bg-gradient-to-r from-purple-500/20 to-pink-500/20 rounded-xl p-4 border border-purple-500/30">
                    <div className="flex items-center justify-between mb-3">
                      <span className="font-medium text-purple-200">GIS Metadata</span>
                      <span className="text-xs text-purple-300 bg-purple-500/20 px-2 py-1 rounded">
                        JSON
                      </span>
                    </div>
                    <div className="text-sm text-purple-300 mb-3">
                      <div>{results.gisData.coordinateSystem}</div>
                      <div>Spatial bounds included</div>
                    </div>
                    <button
                      onClick={() => downloadFile('gis')}
                      className="w-full bg-purple-600 text-white py-2 px-4 rounded-lg hover:bg-purple-700 transition-colors flex items-center justify-center font-medium"
                    >
                      <FileText className="w-4 h-4 mr-2" />
                      Download Metadata
                    </button>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Enhanced Depth Map Modal */}
      {showDepthModal && (results.depthMap || uploadedImage) && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
          <div className="absolute inset-0 bg-black/80 backdrop-blur-sm" onClick={() => setShowDepthModal(false)} />
          <div className="relative bg-slate-800 rounded-2xl p-6 max-w-6xl max-h-[90vh] overflow-auto">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-xl font-semibold">
                Interactive Depth Analysis
              </h3>
              <button 
                onClick={() => setShowDepthModal(false)}
                className="p-2 bg-slate-700 rounded-lg hover:bg-slate-600 transition-colors"
              >
                ✕
              </button>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {uploadedImage && (
                <div>
                  <h4 className="text-sm font-medium text-slate-300 mb-2">Original Image</h4>
                  <img
                    src={uploadedImage.url}
                    alt="Original"
                    className="w-full rounded-lg"
                  />
                </div>
              )}
              {results.depthMap && (
                <div>
                  <h4 className="text-sm font-medium text-slate-300 mb-2">Interactive Depth Map</h4>
                  <div className="relative rounded-lg overflow-hidden border border-slate-600">
                    <div
                      className="w-full cursor-move"
                      style={{
                        transform: `translate(${depthPan.x}px, ${depthPan.y}px) scale(${depthZoom})`,
                        transformOrigin: 'center center'
                      }}
                      onWheel={handleDepthWheel}
                      onMouseDown={handleDepthMouseDown}
                      onMouseMove={handleDepthMouseMove}
                      onMouseUp={handleDepthMouseUp}
                      onMouseLeave={handleDepthMouseUp}
                    >
                      <img
                        src={results.depthMap}
                        alt="Depth Map"
                        className="w-full pointer-events-none"
                        draggable={false}
                      />
                    </div>
                    <div className="absolute bottom-2 left-2 right-2 flex justify-between items-center">
                      <div className="bg-black/80 text-white px-2 py-1 rounded text-xs backdrop-blur-sm">
                        Zoom: {(depthZoom * 100).toFixed(0)}%
                      </div>
                      <div className="flex space-x-1">
                        <button
                          onClick={() => setDepthZoom(prev => Math.min(prev * 1.2, 10))}
                          className="p-1 bg-black/80 text-white rounded text-xs hover:bg-black/90"
                        >
                          <ZoomIn className="w-3 h-3" />
                        </button>
                        <button
                          onClick={() => setDepthZoom(prev => Math.max(prev * 0.8, 0.5))}
                          className="p-1 bg-black/80 text-white rounded text-xs hover:bg-black/90"
                        >
                          <ZoomOut className="w-3 h-3" />
                        </button>
                        <button
                          onClick={resetDepthView}
                          className="p-1 bg-black/80 text-white rounded text-xs hover:bg-black/90"
                        >
                          <RotateCcw className="w-3 h-3" />
                        </button>
                      </div>
                    </div>
                  </div>
                  <div className="mt-2 text-xs text-slate-400 text-center">
                    🖱️ Drag to pan • ⚡ Wheel to zoom • Colors indicate depth
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Settings Modal */}
      {showSettingsModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
          <div className="absolute inset-0 bg-black/80 backdrop-blur-sm" onClick={() => setShowSettingsModal(false)} />
          <div className="relative bg-slate-800 rounded-2xl p-6 w-full max-w-md max-h-[90vh] overflow-auto">
            <div className="flex items-center justify-between mb-6">
              <h3 className="text-xl font-semibold">Viewer Settings</h3>
              <button 
                onClick={() => setShowSettingsModal(false)}
                className="p-2 bg-slate-700 rounded-lg hover:bg-slate-600 transition-colors"
              >
                ✕
              </button>
            </div>
            
            <div className="space-y-6">
              <div>
                <label className="block text-sm font-medium text-slate-300 mb-3">Output Format</label>
                <div className="grid grid-cols-2 gap-2">
                  {['las', 'ply', 'xyz', 'laz', 'mesh_ply'].map(format => (
                    <button
                      key={format}
                      onClick={() => setSettings(prev => ({ ...prev, outputFormat: format }))}
                      className={`py-2 px-3 text-sm rounded-lg border transition-all ${
                        settings.outputFormat === format 
                          ? 'border-blue-500 bg-blue-500/20 text-blue-200' 
                          : 'border-slate-600 text-slate-300 hover:border-slate-500'
                      }`}
                    >
                      {format.toUpperCase()}
                    </button>
                  ))}
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium text-slate-300 mb-3">Point Density</label>
                <div className="flex space-x-2">
                  {['low', 'medium', 'high'].map(density => (
                    <button
                      key={density}
                      onClick={() => setSettings(prev => ({ ...prev, pointDensity: density }))}
                      className={`flex-1 py-2 px-3 text-sm rounded-lg transition-all ${
                        settings.pointDensity === density 
                          ? 'bg-green-600 text-white' 
                          : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
                      }`}
                    >
                      {density}
                    </button>
                  ))}
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium text-slate-300 mb-2">Point Size: {pointSize}</label>
                <input 
                  type="range" 
                  min="1" 
                  max="10" 
                  step="1" 
                  value={pointSize} 
                  onChange={(e) => setPointSize(Number(e.target.value))} 
                  className="w-full accent-blue-500" 
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-slate-300 mb-2">Depth Scale: {settings.depthScale}</label>
                <input 
                  type="range" 
                  min="5" 
                  max="50" 
                  step="1" 
                  value={settings.depthScale} 
                  onChange={(e) => setSettings(prev => ({ ...prev, depthScale: Number(e.target.value) }))} 
                  className="w-full accent-blue-500" 
                />
              </div>

              <div className="grid grid-cols-2 gap-4">
                <label className="flex items-center gap-2 text-sm text-slate-300">
                  <input 
                    type="checkbox" 
                    checked={showGrid} 
                    onChange={(e) => setShowGrid(e.target.checked)}
                    className="accent-blue-500"
                  />
                  Show Grid
                </label>
                <label className="flex items-center gap-2 text-sm text-slate-300">
                  <input 
                    type="checkbox" 
                    checked={showAxes} 
                    onChange={(e) => setShowAxes(e.target.checked)}
                    className="accent-blue-500"
                  />
                  Show Axes
                </label>
                <label className="flex items-center gap-2 text-sm text-slate-300">
                  <input 
                    type="checkbox" 
                    checked={wireframe} 
                    onChange={(e) => setWireframe(e.target.checked)}
                    className="accent-blue-500"
                  />
                  Wireframe
                </label>
                <label className="flex items-center gap-2 text-sm text-slate-300">
                  <input 
                    type="checkbox" 
                    checked={cameraAutoRotate} 
                    onChange={(e) => setCameraAutoRotate(e.target.checked)}
                    className="accent-blue-500"
                  />
                  Auto Rotate
                </label>
              </div>
            </div>
            
            <div className="mt-8 flex justify-end">
              <button 
                onClick={() => setShowSettingsModal(false)} 
                className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default App;