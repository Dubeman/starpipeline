import gym
import numpy as np
import time
from picamera2 import Picamera2
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import cv2
from astrometry_handler import build_cmd, run

class CameraEnv(gym.Env):
    def __init__(self):
        super().__init__()

        self.picam2 = Picamera2()
        config = self.picam2.create_still_configuration()
        self.picam2.configure(config)
        self.picam2.start()
        time.sleep(2)

        # Exposure/gain boundaries
        self.min_exposure = 100_000   # 0.1s
        self.max_exposure = 10_000_000 # 10s
        self.min_gain = 1.0
        self.max_gain = 16.0

        # RL tuning range (deltas for exp,gain)
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, -1.0]),
            high=np.array([1.0, 1.0]),
            dtype=np.float32
        )

        self.observation_space = gym.spaces.Box(
            low=np.array([0.0, self.min_exposure, self.min_gain]),
            high=np.array([255.0, self.max_exposure, self.max_gain]),
            dtype=np.float32
        )

        self.exposure = self.min_exposure
        self.gain = self.min_gain

        self.current_step = 0
        self.max_steps = 10

        self.session = 0
        self.cap = 0

    def warm_start(self):
        print("Running warm start...")
        exposure = 2_000_000
        gain = 2.0
        while exposure <= self.max_exposure:
            self.picam2.set_controls({
                "AeEnable": False,
                "ExposureTime": int(exposure),
                "AnalogueGain": float(gain)
            })
            time.sleep(0.2)
            frame = self.picam2.capture_array()
            if self._is_solvable(frame):
                print(f"[WARM START] Solved at {exposure}us, gain={gain}")
                self.exposure = exposure
                self.gain = gain
                return
            else:
                print(f"[WARM START] Failed at {exposure}us, increasing")
                exposure *= 2
        print("[WARM START] Max exposure reached without solving.")
        self.exposure = self.max_exposure
        self.gain = self.min_gain

    def reset(self):
        self.warm_start()
        self.current_step = 0
        return np.array([self._capture_brightness(), self.exposure, self.gain], dtype=np.float32)

    def step(self, action):
        max_exposure_delta = 2_000_000   # 2 seconds
        max_gain_delta = 4.0

        # Actions from [-1, 1], scale with meaningful units
        delta_exp = float(action[0]) * max_exposure_delta
        delta_gain = float(action[1]) * max_gain_delta

        self.exposure = int(np.clip(self.exposure + delta_exp, self.min_exposure, self.max_exposure))
        self.gain = float(np.clip(self.gain + delta_gain, self.min_gain, self.max_gain))

        self.picam2.set_controls({
            "AeEnable": False,
            "ExposureTime": self.exposure,
            "AnalogueGain": self.gain
        })
        time.sleep(0.2)
        print(f"Trying capture {self.exposure}us ({self.gain})")
        frame = self.picam2.capture_array()

        brightness = frame.mean()
        obs = np.array([brightness, self.exposure, self.gain], dtype=np.float32)

        # Solvability-based reward
        if self._is_solvable(frame):
            reward = 100.0
        else:
            reward = -100.0
        reward -= 0.0001 * self.exposure + self.gain
        self.current_step += 1
        done = self.current_step >= self.max_steps
        if reward > 0:
            print("Solved")
        else:
            print("Not Solved")
        return obs, reward, done, {}

    def _is_solvable(self, image):
        name = f"apr30/{self.session}_{self.cap}.jpg"
        self.cap+=1
        cv2.imwrite(name, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

        cmd = build_cmd(name, {"limit": str(10), "lower_scale": "19", "upper_scale": "20", "downsample": str(4)})
        res = run(cmd)

        solved = res[1] != "Failed"

        coords = res[1]

        return solved

    def _capture_brightness(self):
        frame = self.picam2.capture_array()
        return frame.mean()

    def close(self):
        self.picam2.stop()

env = DummyVecEnv([lambda: CameraEnv()])
model = PPO("MlpPolicy", env, verbose=2)
model.learn(total_timesteps=10)
model.save("exp_gain_agent")
env.close()

 