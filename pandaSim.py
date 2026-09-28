"""Demo: el Franka Emika Panda (MuJoCo Menagerie) sigue tu mano y el pellizco abre y cierra la pinza.

Uso (primero el simulador, luego handDetector.py desde PyCharm):
    mjpython pandaSim.py --filter oneeuro

El experimento de filtros (tools/replay.py) usa el brazo simple de armSim.py; este script es el demo visual.
Reutiliza el mismo mensaje UDP, el mismo mapeo de profundidad y los mismos filtros.

Control: el objetivo filtrado se convierte en ángulos con una cinemática inversa de 6 grados (posición de la
punta y pinza apuntando hacia abajo) por mínimos cuadrados amortiguados. La séptima articulación sobra, así
que en el espacio nulo se empuja hacia la postura "home" para evitar posturas raras. Los servos del modelo
siguen esos ángulos, con compensación de gravedad como el controlador del Panda real.
"""
import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

import armSim
import filters

SCENE = Path(__file__).resolve().parent / "models" / "franka_emika_panda" / "scene.xml"
TCP_OFFSET = 0.1034  # punta de la pinza respecto al cuerpo "hand", m (TCP estándar del Franka Hand)
ARM_BODIES = ("link1", "link2", "link3", "link4", "link5", "link6", "link7", "hand", "left_finger", "right_finger")

# Mismo convenio que armSim.hand_to_target, escalado al espacio de trabajo del Panda (m, marco de la base).
# Con la pinza hacia abajo, la zona lejana y alta (x >= 0.55 con z >= 0.50) queda fuera de alcance: el
# rango se eligió para que todas las esquinas queden con menos de 5 mm de error de la IK.
X_RANGE = (0.60, 0.35)   # mano a 30 cm de la cámara -> adelante; a 65 cm -> atrás
Y_RANGE = (0.30, -0.30)  # izquierda y derecha
Z_RANGE = (0.45, 0.12)   # arriba y abajo
PINCH_CLOSE_CM, PINCH_OPEN_CM = 2.5, 4.5  # histéresis: cierra con el pellizco bajo 2.5 cm, abre sobre 4.5 cm
PINCH_FRAMES = 3         # frames seguidos para cambiar (~100 ms): un frame mal detectado no cierra la pinza
GRIPPER_OPEN = 255.0     # rango del actuador de la pinza en el modelo de Menagerie: 0 cerrada, 255 abierta


def build_model():
    """Escena de Menagerie sin tocar sus archivos: se agregan la punta, la compensación de gravedad y los marcadores."""
    spec = mujoco.MjSpec.from_file(str(SCENE))
    spec.body("hand").add_site(name="tcp", pos=[0, 0, TCP_OFFSET], size=[0.008, 0, 0], rgba=[1, 1, 0, 1])
    for name in ARM_BODIES:
        spec.body(name).gravcomp = 1.0
    for name, size, rgba in (("target", 0.02, [1, 0, 0, 0.6]), ("raw", 0.012, [0.5, 0.5, 0.5, 0.5])):
        body = spec.worldbody.add_body(name=name, mocap=True, pos=[0.55, 0, 0.52])
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[size, 0, 0], rgba=rgba, contype=0, conaffinity=0)
    return spec.compile()


def hand_to_target(msg):
    dist_cm = armSim.DEPTH_K / msg["d"]
    return np.array([armSim.lin(dist_cm, 30, 65, *X_RANGE),
                     armSim.lin(msg["u"], 0.20, 0.80, *Y_RANGE),
                     armSim.lin(msg["v"], 0.20, 0.80, *Z_RANGE)])


class Gripper:
    """Pinza abierta o cerrada según el pellizco, con histéresis y un mínimo de frames seguidos para cambiar."""

    def __init__(self):
        self.closed = False
        self.count = 0

    def update(self, pinch_m):
        cm = pinch_m * 100
        flip = cm > PINCH_OPEN_CM if self.closed else cm < PINCH_CLOSE_CM
        self.count = self.count + 1 if flip else 0
        if self.count >= PINCH_FRAMES:
            self.closed, self.count = not self.closed, 0
        return 0.0 if self.closed else GRIPPER_OPEN


class PandaIK:
    """Cinemática inversa de 6 grados: mínimos cuadrados amortiguados y postura preferida en el espacio nulo."""

    def __init__(self, model, damping=0.05, max_step=0.2, null_gain=0.3, iters=30):
        self.m, self.d = model, mujoco.MjData(model)
        self.site = model.site("tcp").id
        self.damping, self.max_step, self.null_gain, self.iters = damping, max_step, null_gain, iters
        self.lo, self.hi = model.jnt_range[:7, 0], model.jnt_range[:7, 1]
        home = model.key("home")
        self.q_home = home.qpos[:7].copy()
        self.d.qpos[:] = home.qpos
        mujoco.mj_kinematics(model, self.d)
        self.R_des = self.d.site_xmat[self.site].reshape(3, 3).copy()  # pinza hacia abajo, como en "home"
        self.jacp, self.jacr = np.zeros((3, model.nv)), np.zeros((3, model.nv))

    def solve(self, target, q_init):
        q = q_init.copy()
        for _ in range(self.iters):
            self.d.qpos[:7] = q
            mujoco.mj_kinematics(self.m, self.d)
            mujoco.mj_comPos(self.m, self.d)
            R = self.d.site_xmat[self.site].reshape(3, 3)
            err = np.concatenate([target - self.d.site_xpos[self.site],
                                  0.5 * np.cross(R.T, self.R_des.T).sum(axis=0)])  # error de orientación
            if np.linalg.norm(err[:3]) < 1e-4 and np.linalg.norm(err[3:]) < 1e-3:
                break
            mujoco.mj_jacSite(self.m, self.d, self.jacp, self.jacr, self.site)
            J = np.vstack([self.jacp[:, :7], self.jacr[:, :7]])
            J_pinv = J.T @ np.linalg.inv(J @ J.T + self.damping ** 2 * np.eye(6))
            dq = J_pinv @ err + (np.eye(7) - J_pinv @ J) @ (self.null_gain * (self.q_home - q))
            n = np.linalg.norm(dq)
            if n > self.max_step:
                dq *= self.max_step / n
            q = np.clip(q + dq, self.lo, self.hi)
        return q


class PandaDemo:
    """Estado del demo. Separado del visor para poder probarlo sin ventana con una sesión grabada."""

    def __init__(self, filt):
        self.m = build_model()
        self.d = mujoco.MjData(self.m)
        mujoco.mj_resetDataKeyframe(self.m, self.d, self.m.key("home").id)  # también pone ctrl en "home", pinza abierta
        mujoco.mj_forward(self.m, self.d)
        self.ik = PandaIK(self.m)
        self.filt = filt
        self.gripper = Gripper()
        self.q_des = self.ik.q_home.copy()
        self.target_id = self.m.body("target").mocapid[0]
        self.raw_id = self.m.body("raw").mocapid[0]

    def on_message(self, msg):
        raw = hand_to_target(msg)
        target = self.filt.update(raw, msg["ts"] / 1000)
        self.q_des = self.ik.solve(target, self.q_des)
        self.d.ctrl[:7] = self.q_des
        if "p" in msg:  # un detector anterior no manda el pellizco: la pinza se queda como está
            self.d.ctrl[7] = self.gripper.update(msg["p"])
        self.d.mocap_pos[self.target_id] = target
        self.d.mocap_pos[self.raw_id] = raw
        return target

    def step_to(self, t):
        """Simula hasta el tiempo t (s de simulación)."""
        while self.d.time < t:
            mujoco.mj_step(self.m, self.d)


def main():
    parser = argparse.ArgumentParser(description="Demo del Franka Panda controlado por la mano. Correr con mjpython.")
    filters.add_arguments(parser)
    args = parser.parse_args()
    params = filters.params_from_arguments(args)
    demo = PandaDemo(filters.make_filter(args.filter, **params))
    print(f"Filtro: {args.filter} {params}")

    sock = armSim.make_receiver()
    with mujoco.viewer.launch_passive(demo.m, demo.d) as viewer:
        viewer.cam.azimuth, viewer.cam.elevation, viewer.cam.distance = 140, -20, 1.9
        viewer.cam.lookat[:] = [0.45, 0.0, 0.35]
        start = time.perf_counter() - demo.d.time  # la simulación avanza al ritmo del reloj real
        while viewer.is_running():
            msg = armSim.read_latest(sock)
            with viewer.lock():
                if msg is not None:
                    demo.on_message(msg)
                demo.step_to(time.perf_counter() - start)
            viewer.sync()
            time.sleep(1 / 120)


if __name__ == "__main__":
    main()
