from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import numpy as np
from core.env import CR3BPHaloEnv
from core.model import PPO_GAE
import torch
import asyncio
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

orbit_path = "core/HaloorbitManifoldsL2.txt"
model_path = "weights/model.pth"

dummy_env = CR3BPHaloEnv(filepath=orbit_path)

state_dim = dummy_env.observation_space.shape[0]
action_dim = dummy_env.action_space.shape[0]
max_action = float(dummy_env.action_space.high[0])
au_km = dummy_env.au_km
vel_bound = 10e-5 / dummy_env.vDim

agent = PPO_GAE(state_dim, action_dim, au_km)
checkpoint = torch.load(model_path, weights_only=True)
agent.policy.load_state_dict(checkpoint)
agent.policy_old.load_state_dict(checkpoint)

agent.policy.eval()
agent.policy_old.eval()

@app.get("/orbit")
def get_reference_orbit():
    return dummy_env.ref_traj.pos_ref.tolist()

@app.websocket("/run")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    env = CR3BPHaloEnv(filepath=orbit_path)
    
    state = await websocket.receive_json()

    r4, r5, r6 = [np.random.uniform(0, vel_bound) * np.random.choice([-1, 1]) for _ in range(3)]
    state.extend([r4, r5, r6])
    state = np.array(state)

    env.state = env.x_n_ref + state
    env.current_time = 0.0
    env.global_step_count = 0

    dist_km = env._get_distance()
    is_awake = bool(dist_km > env.WAKE_UP_DIST)

    sim_config = {"speed": 1.0}

    # BACKGROUND LISTENER: Constantly checks for speed updates from React
    async def listen_for_updates():
        try:
            while True:
                msg = await websocket.receive_json()
                if isinstance(msg, dict) and msg.get("type") == "speed_update":
                    sim_config["speed"] = msg.get("speed", 1.0)
        except WebSocketDisconnect:
            pass
        except Exception as e:
            print(f"Listener error: {e}")

    # Spin up the listener alongside the main loop
    listener_task = asyncio.create_task(listen_for_updates())

    for _ in range(10000):
        current_state = env.state.copy()
        
        # Deadband Logic (Wake up at >20k, Sleep at <5k)
        if is_awake and dist_km < env.SLEEP_DIST:
            is_awake = False
        elif not is_awake and dist_km > env.WAKE_UP_DIST:
            is_awake = True
            
        # Agent Action vs Coasting
        if is_awake:
            # Reconstruct the 21D observation exactly as the agent expects it
            agent_obs = env._get_obs()
            
            # Fetch deterministic action [-1.0, 1.0]
            raw_action = agent.select_action(agent_obs, evaluate=True)
            
            # Scale the thrust to required range
            applied_thrust = raw_action * max_action
        else:
            # Apply 0 thrust
            applied_thrust = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        main_list = current_state.tolist()
        thrust_list = applied_thrust.tolist()
        main_list.extend(thrust_list)
        main_list.append(dist_km)
        try:
            await websocket.send_json(main_list)
        except WebSocketDisconnect:
            break
        
        # Step the physics manually by 1 dt tick
        env._step_physics(applied_thrust)
        dist_km = env._get_distance()
        
        # Terminal condition
        if dist_km > 100000:
            break

        await asyncio.sleep(0.016/sim_config["speed"])

    listener_task.cancel()
    try:
        await websocket.close()
    except Exception:
        pass