import React, { useState, useEffect, useRef } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Stars, Line, Sphere } from '@react-three/drei';
import * as THREE from 'three';

// 3D SCENE COMPONENT
const SimulationScene = ({ satelliteRef, trailRef, orbitPoints }) => {
  const satMeshRef = useRef();
  
  useFrame(() => {
    if (satMeshRef.current && satelliteRef.current) {
      satMeshRef.current.position.set(...satelliteRef.current);
    }
  });

  return (
    <>
      <ambientLight intensity={0.2} />
      <pointLight position={[50, 50, 50]} intensity={1.5} />
      <Stars radius={150} depth={50} count={1500} factor={4} saturation={0} fade speed={1} />
      <OrbitControls makeDefault target={[0, 0, 0]} />

      <Sphere args={[0.08, 16, 16]} position={[0, 0, 0]}>
        <meshBasicMaterial color="#ff0055" wireframe />
      </Sphere>

      <Sphere ref={satMeshRef} args={[0.15, 16, 16]}>
        <meshStandardMaterial color="#00ffcc" emissive="#00ffcc" emissiveIntensity={1.5} />
      </Sphere>

      {trailRef.current.length > 1 && (
        <Line points={trailRef.current} color="#00ffcc" lineWidth={2} transparent opacity={0.6} />
      )}

      {orbitPoints.length > 0 && (
        <Line points={orbitPoints} color="#446688" lineWidth={2} transparent opacity={0.8} />
      )}
    </>
  );
};

// MAIN DASHBOARD COMPONENT
export default function App() {
  const [isRunning, setIsRunning] = useState(false);
  const [errors, setErrors] = useState({ x: 0, y: 0, z: 0 });
  const [orbitPoints, setOrbitPoints] = useState([]);
  const [speed, setSpeed] = useState(1); 
  const [isDocsOpen, setIsDocsOpen] = useState(false);
  
  const [telemetry, setTelemetry] = useState({
    thrust: [0, 0, 0],
    thrustersOn: false,
    daysElapsed: 0,
    distance: 0.0
  });

  const [serverStatus, setServerStatus] = useState("waking");

  const wsRef = useRef(null);
  const satelliteRef = useRef([0, 0, 0]);
  const trailRef = useRef([]);
  const ticksRef = useRef(0);
  const offsetRef = useRef([0, 0, 0]); 
  const scaleRef = useRef(1);

useEffect(() => {
    fetch("https://kavan0610-station-keeping-backend.hf.space/orbit")
      .then(res => {
        if (!res.ok) throw new Error("Server returned an error status");
        return res.json();
      })
      .then(data => {
        setServerStatus("ready"); 
        
        if (!data || data.length === 0) return;
        
        let min = [Infinity, Infinity, Infinity], max = [-Infinity, -Infinity, -Infinity];
        data.forEach(p => {
          for(let i=0; i<3; i++) {
            if (p[i] < min[i]) min[i] = p[i];
            if (p[i] > max[i]) max[i] = p[i];
          }
        });
        const cx = (max[0] + min[0]) / 2, cy = (max[1] + min[1]) / 2, cz = (max[2] + min[2]) / 2;
        offsetRef.current = [cx, cy, cz]; 
        let maxDist = 0;
        const centeredData = data.map(p => {
          const shifted = [p[0] - cx, p[1] - cy, p[2] - cz];
          maxDist = Math.max(maxDist, Math.abs(shifted[0]), Math.abs(shifted[1]), Math.abs(shifted[2]));
          return shifted;
        });
        const computedScale = 15 / (maxDist === 0 ? 1 : maxDist); 
        scaleRef.current = computedScale;
        const finalOrbit = centeredData.map(p => new THREE.Vector3(p[0] * computedScale, p[1] * computedScale, p[2] * computedScale));
        
        setOrbitPoints(finalOrbit);
      })
      .catch(err => {
        console.error("FAILED to load orbit data", err);
        setServerStatus("error"); 
      });
  }, []);

  const startSimulation = () => {
    if (wsRef.current) wsRef.current.close();
    
    setIsRunning(true);
    trailRef.current = [];
    ticksRef.current = 0;

    wsRef.current = new WebSocket("wss://kavan0610-station-keeping-backend.hf.space/run");

    wsRef.current.onopen = () => {
      const AU_KM = 149597870.71; 
      const initialState = [errors.x / AU_KM, errors.y / AU_KM, errors.z / AU_KM];
      wsRef.current.send(JSON.stringify(initialState));
      wsRef.current.send(JSON.stringify({ type: "speed_update", speed: speed }));
    };

    wsRef.current.onmessage = (event) => {
      const data = JSON.parse(event.data);
      const cx = offsetRef.current[0], cy = offsetRef.current[1], cz = offsetRef.current[2];
      const s = scaleRef.current; 

      const pos = [(data[0] - cx) * s, (data[1] - cy) * s, (data[2] - cz) * s];
      const thrust = [data[6], data[7], data[8]];
      const exactDistance = data[9] || 0; 
      const isThrusting = Math.abs(thrust[0]) > 0 || Math.abs(thrust[1]) > 0 || Math.abs(thrust[2]) > 0;

      satelliteRef.current = pos;
      trailRef.current.push(new THREE.Vector3(...pos));
      if (trailRef.current.length > 200) trailRef.current.shift();

      ticksRef.current += 1;

      if (ticksRef.current % 5 === 0) {
        setTelemetry({
          thrust: thrust,
          thrustersOn: isThrusting,
          daysElapsed: (ticksRef.current * 0.0902).toFixed(2), 
          distance: parseFloat(exactDistance).toFixed(6) 
        });
      }
    };

    wsRef.current.onclose = () => setIsRunning(false);
  };

  const resetSimulation = () => {
    if (wsRef.current) wsRef.current.close();
    setIsRunning(false);
    setErrors({ x: 0, y: 0, z: 0 });
    setSpeed(1); 
    satelliteRef.current = [0, 0, 0];
    trailRef.current = [];
    ticksRef.current = 0;
    setTelemetry({ thrust: [0, 0, 0], thrustersOn: false, daysElapsed: 0, distance: 0.0 });
  };

  const handleSliderChange = (axis, value) => {
    setErrors(prev => ({ ...prev, [axis]: parseFloat(value) }));
  };

  const handleSpeedChange = (newSpeed) => {
    setSpeed(newSpeed);
    if (isRunning && wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: "speed_update", speed: newSpeed }));
    }
  };

  return (
    <div className="flex h-screen w-full bg-black text-green-400 font-mono overflow-hidden">
      
      {/* 3D CANVAS PORTION */}
      <div className="relative w-3/4 h-full border-r border-green-900/50">
        
        {/* Top Left: Mission Clock */}
        <div className="absolute top-4 left-4 z-10 bg-black/60 border border-green-900 p-2 rounded text-sm backdrop-blur-sm shadow-[0_0_15px_rgba(0,255,100,0.05)]">
          <div className="text-green-500/80 mb-1">MISSION CLOCK</div>
          <div className="text-xl font-bold text-white">T+ {telemetry.daysElapsed} DAYS</div>
        </div>

        {/* Top Right: Docs Button */}
        <button 
          onClick={() => setIsDocsOpen(true)}
          className="absolute top-4 right-4 z-10 bg-black/60 border border-green-900 px-6 py-3 rounded text-sm font-bold tracking-widest text-green-500 hover:bg-green-900/40 hover:text-white backdrop-blur-sm transition-all shadow-[0_0_15px_rgba(0,255,100,0.05)]"
        >
          DOCS
        </button>

        <Canvas camera={{ position: [15, 10, 25], fov: 50 }}>
          <SimulationScene satelliteRef={satelliteRef} trailRef={trailRef} orbitPoints={orbitPoints} />
        </Canvas>
      </div>

      {/* COMPACT TELEMETRY DASHBOARD */}
      <div className="w-1/4 h-full bg-[#050505] p-4 flex flex-col shadow-[-10px_0_30px_rgba(0,255,100,0.05)]">
        
        <h1 className="text-lg font-bold tracking-widest text-white mb-4 border-b border-green-900 pb-2">
          STATION KEEPING
          <span className="block text-[10px] text-green-600 mt-1">CR3BP TELEMETRY UPLINK</span>
        </h1>

        {/* --- SERVER STATUS BANNER --- */}
        {serverStatus === "waking" && (
          <div className="mb-4 bg-yellow-900/20 border border-yellow-500/50 p-2 rounded text-[10px] text-yellow-500 font-bold tracking-widest animate-pulse shadow-[0_0_10px_rgba(234,179,8,0.2)]">
            WAKING UP CLOUD SERVER (MAY TAKE UPTO 60S)...
          </div>
        )}
        {serverStatus === "error" && (
          <div className="mb-4 bg-red-900/20 border border-red-500/50 p-2 rounded text-[10px] text-red-500 font-bold tracking-widest shadow-[0_0_10px_rgba(239,68,68,0.2)]">
            CONNECTION FAILED. PLEASE REFRESH.
          </div>
        )}

        <div className="mb-4 space-y-3">
          <h2 className="text-[10px] text-green-500/80 font-bold tracking-widest">INITIAL PERTURBATION (km)</h2>
          
          {['x', 'y', 'z'].map((axis) => (
            <div key={axis}>
              <div className="flex justify-between text-[10px] mb-0.5">
                <span className="uppercase">ERROR {axis}</span>
                <span>{errors[axis]}</span>
              </div>
              <input 
                type="range" min="-1200" max="1200" value={errors[axis]}
                onChange={(e) => handleSliderChange(axis, e.target.value)}
                disabled={isRunning}
                className="w-full h-1 bg-green-900/30 rounded-lg appearance-none cursor-pointer accent-green-500"
              />
            </div>
          ))}

          <div className="flex gap-3 mt-4">
            <button 
              onClick={startSimulation} 
              disabled={isRunning || serverStatus !== "ready"}
              className={`w-1/2 py-2 text-sm rounded font-bold tracking-wider transition-all duration-300 ${
                (isRunning || serverStatus !== "ready") ? 'bg-green-900/30 text-green-700 cursor-not-allowed' : 'bg-green-600/20 text-green-400 border border-green-500 hover:bg-green-500 hover:text-black hover:shadow-[0_0_10px_rgba(0,255,100,0.4)]'
              }`}
            >
              {isRunning ? 'ACTIVE' : 'INITIATE'}
            </button>
            
            <button 
              onClick={resetSimulation}
              className="w-1/2 py-2 text-sm rounded font-bold tracking-wider transition-all duration-300 bg-red-900/20 text-red-500 border border-red-500/50 hover:bg-red-500 hover:border-red-500 hover:text-white hover:shadow-[0_0_10px_rgba(255,0,0,0.4)]"
            >
              RESET
            </button>
          </div>
        </div>

        <div className="mb-6">
          <h2 className="text-[10px] text-green-500/80 font-bold tracking-widest mb-2 border-b border-green-900/50 pb-1">TIME WARP (PLAYBACK SPEED)</h2>
          <div className="flex gap-2">
            {[1, 2, 3].map((s) => (
              <button
                key={s}
                onClick={() => handleSpeedChange(s)}
                className={`flex-1 py-1.5 text-xs font-bold rounded border transition-all duration-300 ${
                  speed === s
                  ? 'bg-green-500 text-black border-green-500 shadow-[0_0_10px_rgba(0,255,100,0.3)]'
                  : 'bg-black/50 text-green-600 border-green-900/50 hover:border-green-500 hover:text-green-400'
                }`}
              >
                {s}X
              </button>
            ))}
          </div>
        </div>

        <div className="mt-auto space-y-2">
          <h2 className="text-[10px] text-green-500/80 font-bold tracking-widest border-b border-green-900 pb-1">LIVE TELEMETRY</h2>
          
          <div className="flex justify-between items-center bg-black/50 p-2 border border-green-900/50 rounded">
            <span className="text-xs">THRUSTERS</span>
            <span className={`px-2 py-0.5 text-[10px] font-bold rounded ${telemetry.thrustersOn ? 'bg-red-500/20 text-red-500 border border-red-500/50 shadow-[0_0_8px_rgba(255,0,0,0.3)]' : 'bg-green-900/20 text-green-700'}`}>
              {telemetry.thrustersOn ? 'ACTIVE' : 'STANDBY'}
            </span>
          </div>

          <div className="bg-black/50 p-2 border border-green-900/50 rounded space-y-1">
            <div className="text-[10px] text-green-600 mb-1">THRUST VECTOR (kN)</div>
            <div className="flex justify-between text-xs"><span>Tx:</span> <span>{telemetry.thrust[0].toFixed(5)}</span></div>
            <div className="flex justify-between text-xs"><span>Ty:</span> <span>{telemetry.thrust[1].toFixed(5)}</span></div>
            <div className="flex justify-between text-xs"><span>Tz:</span> <span>{telemetry.thrust[2].toFixed(5)}</span></div>
          </div>

          <div className="bg-black/50 p-2 border border-green-900/50 rounded flex flex-col">
            <span className="text-[10px] text-green-600 mb-0.5">DEVIATION DISTANCE</span>
            <span className="text-lg font-bold text-white">{telemetry.distance} <span className="text-[10px] text-green-600 font-normal">KM</span></span>
          </div>
        </div>
      </div>

      {/* --- FULL SCREEN DOCS MODAL --- */}
      {isDocsOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-md p-8 sm:p-12">
          
          {/* Modal Container */}
          <div className="relative w-full max-w-5xl h-full max-h-[90vh] overflow-y-auto bg-[#050505] border border-green-900/50 rounded-lg shadow-[0_0_50px_rgba(0,255,100,0.1)] p-8 sm:p-12 text-green-400 font-mono hide-scrollbar">
            
            {/* Close Button */}
            <button 
              onClick={() => setIsDocsOpen(false)}
              className="absolute top-6 right-6 text-green-600 hover:text-white text-3xl font-bold leading-none transition-colors"
              aria-label="Close Documentation"
            >
              &times;
            </button>

            {/* Docs Content */}
            <div className="max-w-3xl mx-auto space-y-10">
              
              <div className="border-b border-green-900 pb-6">
                <h2 className="text-3xl font-bold text-white tracking-widest mb-2">PROJECT DOCUMENTATION</h2>
                <p className="text-green-500/80 text-sm">Autonomous Station Keeping via Reinforcement Learning</p>
              </div>

              <section className="space-y-4">
                <h3 className="text-xl font-bold text-white border-l-4 border-green-500 pl-3">1. The Physics (CR3BP Environment)</h3>
                <p className="text-sm leading-relaxed text-green-100/80">
                  This simulation utilizes the Circular Restricted Three-Body Problem (CR3BP) to model the gravitational dynamics of a spacecraft near the Sun-Earth L2 Lagrange point. Because L2 is an unstable equilibrium point, a spacecraft will naturally drift away over time.
                </p>
                <ul className="list-disc list-inside text-sm text-green-100/80 space-y-2 ml-2">
                  <li><strong>State Space:</strong> 21-Dimensional vector including absolute state, local positional/velocity errors, and spatial lookahead breadcrumbs (10%, 25%, 50% of orbit period).</li>
                  <li><strong>Action Space:</strong> Continuous 3D thrust vector scaled to physical engine limits.</li>
                  <li><strong>Deadband Logic:</strong> The agent enters a "sleep" coasting mode when deviation is &lt; 5,000 km, and wakes up to apply corrective thrust when deviation exceeds 5,000 km.</li>
                </ul>
                
                <div className="mt-6 flex flex-col items-center border border-green-900/50 rounded-lg p-2 bg-black/50">
                    <p className="text-xs text-green-600 mb-2 uppercase tracking-widest">CR3BP Rotating Frame & Lagrange Points</p>
                    <img 
                      src="/Lagrange_Contours-1.jpeg" 
                      alt="L2 Point Grvitational Field" 
                      className="max-w-full h-auto rounded"
                    />
                </div>
              </section>

              <section className="space-y-4">
                <h3 className="text-xl font-bold text-white border-l-4 border-green-500 pl-3">2. The AI (Proximal Policy Optimization)</h3>
                <p className="text-sm leading-relaxed text-green-100/80">
                  The spacecraft is piloted entirely by a trained neural network. It was trained using PPO (Proximal Policy Optimization) with Generalized Advantage Estimation (GAE).
                </p>
                <p className="text-sm leading-relaxed text-green-100/80">
                  Instead of relying on hardcoded orbital mechanics formulas for fuel-optimal burns, the agent learned to balance the penalty of drifting away from the reference orbit against the penalty of consuming propellant. 
                </p>
                
                <div className="bg-green-900/10 border border-green-900/50 p-4 rounded mt-4">
                  <h4 className="text-xs font-bold text-green-500 mb-2">HYPERPARAMETERS</h4>
                  <div className="grid grid-cols-2 gap-2 text-xs text-white">
                    <div>Max Global Steps: <span className="text-green-400">10,000 (~5 Orbits)</span></div>
                    <div>Canonical dt: <span className="text-green-400">~2.16 Hours</span></div>
                  </div>
                </div>
              </section>

              <section className="space-y-4">
                <h3 className="text-xl font-bold text-white border-l-4 border-green-500 pl-3">3. System Architecture</h3>
                <p className="text-sm leading-relaxed text-green-100/80">
                  This dashboard is a decoupled full-stack application designed to run inference in real-time.
                </p>
                <ul className="list-disc list-inside text-sm text-green-100/80 space-y-2 ml-2">
                  <li><strong>Backend:</strong> A FastAPI server orchestrating the PyTorch inference and the Gymnasium physics engine.</li>
                  <li><strong>Communication:</strong> A bidirectional WebSocket streams 60 FPS state vectors to the client, while asynchronously listening for user playback speed commands.</li>
                  <li><strong>Frontend:</strong> React, Three.js, and React Three Fiber (R3F) handling the high-performance 3D rendering and UI state.</li>
                </ul>
              </section>

              {/* GitHub Link Button */}
              <div className="pt-8 border-t border-green-900 flex justify-center">
                <a 
                  href="https://github.com/kavan0610/Autonomous-Station-Keeping-Using-RL" 
                  target="_blank" 
                  rel="noopener noreferrer"
                  className="bg-green-600/20 text-green-400 border border-green-500 px-8 py-3 rounded font-bold tracking-wider hover:bg-green-500 hover:text-black transition-all shadow-[0_0_15px_rgba(0,255,100,0.2)]"
                >
                  VIEW SOURCE ON GITHUB
                </a>
              </div>

            </div>
          </div>
        </div>
      )}

    </div>
  );
}