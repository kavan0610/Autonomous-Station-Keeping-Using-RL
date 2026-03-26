import numpy as np
import gymnasium as gym
from gymnasium import spaces
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize

class ReferenceTrajectory:
    """
    Handles the continuous mathematical representation of the target orbit
    and calculates the exact shortest geometric distance and state errors.
    """
    def __init__(self, filepath="HaloorbitManifoldsL2.txt"):
        try:
            data = np.loadtxt(filepath)
        except OSError:
            print("WARNING: Trajectory file not found. Generating dummy reference trajectory for testing.")
            t = np.linspace(0, 3.14, 100)
            data = np.column_stack((t, np.cos(t), np.sin(t), np.zeros_like(t)))

        self.t_ref = data[:, 0]
        self.pos_ref = data[:, 1:4] 
        
        self.dt = 0.0015517606879168 
        self.pos_ref[-1] = self.pos_ref[0]
        
        self.spline = CubicSpline(self.t_ref, self.pos_ref, bc_type='periodic')
        self.period = self.t_ref[-1]
        
    def get_continuous_nearest_state(self, current_pos_3d):
        """
        Returns the shortest distance, the exact reference position, 
        the reference velocity, and the optimal time on the spline.
        """
        diffs = self.pos_ref - current_pos_3d
        closest_idx = np.argmin(np.linalg.norm(diffs, axis=1))
        t_guess = self.t_ref[closest_idx]
        
        def distance_squared(t):
            t_wrapped = t[0] % self.period 
            return np.sum((self.spline(t_wrapped) - current_pos_3d)**2)
        
        bounds = [(t_guess - self.dt, t_guess + self.dt)]
        res = minimize(distance_squared, x0=[t_guess], bounds=bounds, method='L-BFGS-B')
        
        shortest_distance_nd = np.sqrt(res.fun)
        optimal_t = res.x[0] % self.period
        
        # Get position and velocity (1st derivative) at the optimal time
        ref_pos = self.spline(optimal_t)
        ref_vel = self.spline(optimal_t, 1) 
        
        return shortest_distance_nd, ref_pos, ref_vel, optimal_t
    
class CR3BPHaloEnv(gym.Env):
    """
    Gymnasium Environment for CR3BP Station Keeping.
    Features a 21D observation space with spatial lookahead breadcrumbs.
    """
    def __init__(self, filepath="HaloorbitManifoldsL2.txt"):
        super(CR3BPHaloEnv, self).__init__()
        
        # System Constants
        self.mu = 3.040357143e-06
        self.dt = 0.0015517606879168
        self.au_m = 1.4959787071e11       
        self.au_km = 1.4959787071e8       
        self.vDim = 29785.2543675826      
        self.tDim = 1.9909885711e-7       
        
        self.ref_traj = ReferenceTrajectory(filepath)
        
        self.x_n_ref = np.array([
            1.0112693975423970, 0.0, -0.0008146977347610, 
            0.0, -0.0090501122851750, 0.0
        ])
        
        # RL SPACES
        # Observation Space: 21D Vector 
        # [State (6), Nearest Error (3), Vel Error (3), Lookahead 10% (3), Lookahead 25% (3), Lookahead 50% (3)]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(21,), dtype=np.float32)
        
        max_thrust = 0.01
        self.action_space = spaces.Box(low=-max_thrust, high=max_thrust, shape=(3,), dtype=np.float32)
        
        # Deadband Thresholds
        self.WAKE_UP_DIST = 5000.0
        self.SLEEP_DIST = 5000.0
        
        # Global Simulation State
        self.state = None
        self.current_time = 0.0
        self.global_step_count = 0
        self.max_global_steps = 19990 # 10 full orbits
        self.needs_global_reset = True 

    def _get_obs(self):
        current_pos = self.state[0:3]
        current_vel = self.state[3:6]
        
        # 1. Get the exact target state on the reference trajectory
        _, ref_pos, ref_vel, t_opt = self.ref_traj.get_continuous_nearest_state(current_pos)
        
        # 2. Immediate Spatial and Velocity Errors
        pos_error = ref_pos - current_pos
        vel_error = ref_vel - current_vel
        
        # 3. Calculate Spatial Lookahead Points (10%, 25%, 50%)
        period = self.ref_traj.period
        t_lookahead_1 = (t_opt + period * 0.10) % period
        t_lookahead_2 = (t_opt + period * 0.25) % period
        t_lookahead_3 = (t_opt + period * 0.50) % period
        
        pos_ahead_1 = self.ref_traj.spline(t_lookahead_1)
        pos_ahead_2 = self.ref_traj.spline(t_lookahead_2)
        pos_ahead_3 = self.ref_traj.spline(t_lookahead_3)
        
        # Relative vectors to future points
        vec_ahead_1 = pos_ahead_1 - current_pos
        vec_ahead_2 = pos_ahead_2 - current_pos
        vec_ahead_3 = pos_ahead_3 - current_pos
        
        # 4. Assemble the 21D Observation
        obs = np.concatenate((
            self.state,     # 6D: Absolute State
            pos_error,      # 3D: Local Pos Error
            vel_error,      # 3D: Local Vel Error
            vec_ahead_1,    # 3D: Vector to 10% ahead
            vec_ahead_2,    # 3D: Vector to 25% ahead
            vec_ahead_3     # 3D: Vector to 50% ahead
        ))
        
        return obs.astype(np.float32)

    def _fast_forward_coasting(self):
        coasting_steps = 0
        zero_action = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        
        while True:
            dist_km = self._get_distance()
            
            if dist_km >= self.WAKE_UP_DIST or self.global_step_count >= self.max_global_steps or dist_km > 100000:
                break
                
            self._step_physics(zero_action)
            coasting_steps += 1
            
        return coasting_steps

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
            
        if self.needs_global_reset:
            A = np.array([9.5, 95, 950, 9950, 99500, 495000]) 
            B = np.array([10.5, 105, 1050, 10050, 100500, 500500]) 
            a, b = A[2] / (self.au_m / 1000), B[2] / (self.au_m / 1000)
            
            r1, r2, r3 = [np.random.uniform(a, b) * np.random.choice([-1, 1]) for _ in range(3)]
            vel_bound = 10e-5 / self.vDim
            r4, r5, r6 = [np.random.uniform(0, vel_bound) * np.random.choice([-1, 1]) for _ in range(3)]
            
            self.state = self.x_n_ref + np.array([r1, r2, r3, r4, r5, r6])
            self.current_time = 0.0
            self.global_step_count = 0
            self.needs_global_reset = False
            
            if self._get_distance() < self.WAKE_UP_DIST:
                self._fast_forward_coasting()
        else:
            pass

        info = {
            "initial_distance": self._get_distance(),
            "global_step_count": self.global_step_count
        }
        
        return self._get_obs(), info

    def _equations_of_motion(self, t, state, U):
        x, y, z, xd, yd, zd = state
        r1 = np.sqrt((x + self.mu)**2 + y**2 + z**2)
        r2 = np.sqrt((x - 1 + self.mu)**2 + y**2 + z**2)
        
        Ux_pot = x - ((1 - self.mu) * (x + self.mu) / r1**3) - (self.mu * (x - 1 + self.mu) / r2**3)
        Uy_pot = y - ((1 - self.mu) * y / r1**3) - (self.mu * y / r2**3)
        Uz_pot = - ((1 - self.mu) * z / r1**3) - (self.mu * z / r2**3)
        
        return [xd, yd, zd, 2 * yd + Ux_pot + U[0], -2 * xd + Uy_pot + U[1], Uz_pot + U[2]]

    def _get_distance(self):
        dist_nd, _, _, _ = self.ref_traj.get_continuous_nearest_state(self.state[0:3])
        return dist_nd * self.au_km

    def _step_physics(self, U):
        sol = solve_ivp(
            fun=self._equations_of_motion, 
            t_span=(self.current_time, self.current_time + self.dt), 
            y0=self.state, args=(U,), method='RK45', rtol=1e-10, atol=1e-12
        )
        self.state = sol.y[:, -1]
        self.current_time += self.dt
        self.global_step_count += 1

    def step(self, action):
        self._step_physics(action)
        dist_km = self._get_distance()
        
        mag_U = np.linalg.norm(action)
        step_delta_v_mps = mag_U * self.dt * self.vDim
        reward = -(dist_km / 100000.0) - (step_delta_v_mps * 5.0)
        
        terminated = False
        truncated = False
        
        if self.global_step_count >= self.max_global_steps:
            reward += 1000
            truncated = True
            self.needs_global_reset = True
            
        elif dist_km > 100000:
            terminated = True
            reward -= 1000.0
            self.needs_global_reset = True
            
        elif dist_km < self.SLEEP_DIST:
            terminal_obs = self._get_obs()
            coasting_steps = self._fast_forward_coasting()
            
            reward += 1000.0 + (coasting_steps * 10)
            terminated = True
            
            if self.global_step_count >= self.max_global_steps or self._get_distance() > 100000:
                self.needs_global_reset = True
            else:
                self.needs_global_reset = False
                
            info = {
                'distance_km': dist_km,
                'delta_v_mps': step_delta_v_mps,
                'coasting_duration_steps': coasting_steps,
                'global_step_count': self.global_step_count
            }
            return terminal_obs, reward, terminated, truncated, info

        info = {
            'distance_km': dist_km,
            'delta_v_mps': step_delta_v_mps,
            'coasting_duration_steps': 0,
            'global_step_count': self.global_step_count
        }
        
        return self._get_obs(), reward, terminated, truncated, info

