"""Genesis side of the cross-check: the same Menagerie Go2 model, PD control, timing, seeds and fall rule as mujoco_engine.py."""
import os, sys, json, argparse
import numpy as np, onnxruntime as ort
import genesis as gs
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from mujoco_engine import ROOT, DT, DECIM, KP, KD, EFFORT, VLIM, ACT_SCALE, POLICY_JOINTS, DEFAULT

def quat_to_R(q):  # q = (w, x, y, z)
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                     [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                     [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])

class GSim:
    def __init__(self, friction=1.0, add_mass=0.0):
        self.scene = gs.Scene(sim_options=gs.options.SimOptions(dt=DT, substeps=1), show_viewer=False)
        self.plane = self.scene.add_entity(gs.morphs.Plane())
        self.robot = self.scene.add_entity(gs.morphs.MJCF(file=os.path.join(ROOT, 'go2', 'go2.xml'), pos=(0, 0, 0.4)))
        self.scene.build()
        r = self.robot
        self.dofs = [r.get_joint(n).dofs_idx_local[0] for n in POLICY_JOINTS]
        r.set_dofs_damping(np.zeros(12), self.dofs)          # matched to the MuJoCo harness (damping off)
        r.set_dofs_kp(np.zeros(12), self.dofs); r.set_dofs_kv(np.zeros(12), self.dofs)  # torque control only
        if friction != 1.0: r.set_friction(friction); self.plane.set_friction(friction)
        if add_mass: r.base_link.set_mass(r.base_link.get_mass() + add_mass)
        self.base_idx = r.base_link.idx
    def reset(self, seed=None):
        r = self.robot
        q = DEFAULT.copy(); v = np.zeros(2)
        if seed is not None:
            g = np.random.default_rng(1000 + seed)
            q += g.uniform(-0.1, 0.1, 12); v = g.uniform(-0.2, 0.2, 2)
        r.set_pos(np.array([0, 0, 0.4])); r.set_quat(np.array([1, 0, 0, 0]))
        r.set_dofs_position(q, self.dofs)
        vel = np.zeros(r.n_dofs); vel[:2] = v
        r.set_dofs_velocity(vel)
        self.last = np.zeros(12)
    def R(self): return quat_to_R(np.asarray(self.robot.get_quat()).reshape(-1)[:4])
    def obs(self, cmd):
        r = self.robot; R = self.R()
        lin = R.T @ np.asarray(r.get_vel()).reshape(-1)[:3]
        ang = R.T @ np.asarray(r.get_ang()).reshape(-1)[:3]
        grav = R.T @ np.array([0, 0, -1.0])
        q = np.asarray(r.get_dofs_position(self.dofs)).reshape(-1) - DEFAULT
        qd = np.asarray(r.get_dofs_velocity(self.dofs)).reshape(-1)
        return np.concatenate([lin, ang, grav, cmd, q, qd, self.last]).astype(np.float32)
    def step(self, action):
        self.last = action; target = DEFAULT + ACT_SCALE * action; r = self.robot
        for _ in range(DECIM):
            q = np.asarray(r.get_dofs_position(self.dofs)).reshape(-1); qd = np.asarray(r.get_dofs_velocity(self.dofs)).reshape(-1)
            tau = KP * (target - q) - KD * qd
            hi = np.clip(EFFORT * (1 - qd / VLIM), 0, EFFORT); lo = np.clip(EFFORT * (-1 - qd / VLIM), -EFFORT, 0)
            r.control_dofs_force(np.clip(tau, lo, hi), self.dofs)
            self.scene.step()
    def fallen(self):
        if self.R()[2, 2] < 0.5: return True
        c = self.robot.get_contacts(with_entity=self.plane)
        la, lb = np.asarray(c['link_a']).reshape(-1), np.asarray(c['link_b']).reshape(-1)
        return bool(np.any(la == self.base_idx) or np.any(lb == self.base_idx))
    def kick(self, dvx, dvy):
        v = np.asarray(self.robot.get_dofs_velocity()).reshape(-1).copy(); v[0] += dvx; v[1] += dvy
        self.robot.set_dofs_velocity(v)
    def base(self): return np.asarray(self.robot.get_pos()).reshape(-1)[:3]
