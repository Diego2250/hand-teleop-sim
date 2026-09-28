"""Demo: el Franka Emika Panda (MuJoCo Menagerie) sigue tu mano, con la Shadow Hand o con su pinza.

Uso (primero el simulador, luego handDetector.py desde PyCharm):
    mjpython pandaSim.py --filter oneeuro                  # Shadow Hand: los dedos del robot copian los tuyos
    mjpython pandaSim.py --hand gripper --filter oneeuro   # pinza: el pellizco la abre y la cierra

El experimento de filtros (tools/replay.py) usa el brazo simple de armSim.py; este script es el demo visual.
Reutiliza el mismo mensaje UDP, el mismo mapeo de profundidad y los mismos filtros (--filter filtra el
objetivo del brazo; los dedos se suavizan aparte con una EMA corta).

Brazo: el objetivo filtrado se convierte en ángulos con una cinemática inversa de 6 grados (posición y
orientación de la mano del robot) por mínimos cuadrados amortiguados. La séptima articulación sobra, así que
en el espacio nulo se empuja hacia una postura preferida para evitar posturas raras. Los servos del modelo
siguen esos ángulos, con compensación de gravedad como el controlador del Panda real.

Dedos (Shadow Hand): el doblez de cada articulación de tus dedos, menos su valor con la mano abierta, va a la
articulación equivalente del robot. Es un mapeo articulación a articulación, no un retargeting fino: no busca
que las puntas de los dedos del robot lleguen a la posición exacta de las tuyas.
"""
import argparse
import math
import time
import warnings
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

import armSim
import filters

MODELS = Path(__file__).resolve().parent / "models"
PANDA_SCENE = MODELS / "franka_emika_panda" / "scene.xml"
PANDA_NOHAND = MODELS / "franka_emika_panda" / "panda_nohand.xml"
SHADOW = MODELS / "shadow_hand" / "right_hand.xml"
TCP_OFFSET = 0.1034  # punta de la pinza respecto al cuerpo "hand", m (TCP estándar del Franka Hand)
GRIPPER_BODIES = ("link1", "link2", "link3", "link4", "link5", "link6", "link7", "hand", "left_finger", "right_finger")

# Pinza. Mismo convenio que armSim.hand_to_target, escalado al espacio de trabajo del Panda (m, marco de la
# base). Con la pinza hacia abajo, la zona lejana y alta (x >= 0.55 con z >= 0.50) queda fuera de alcance: el
# rango se eligió para que todas las esquinas queden con menos de 5 mm de error de la IK.
X_RANGE = (0.60, 0.35)   # mano a 30 cm de la cámara -> adelante; a 65 cm -> atrás
Y_RANGE = (0.30, -0.30)  # izquierda y derecha
Z_RANGE = (0.45, 0.12)   # arriba y abajo
PINCH_CLOSE_CM, PINCH_OPEN_CM = 2.5, 4.5  # histéresis: cierra con el pellizco bajo 2.5 cm, abre sobre 4.5 cm
PINCH_FRAMES = 3         # frames seguidos para cambiar (~100 ms): un frame mal detectado no cierra la pinza
GRIPPER_OPEN = 255.0     # rango del actuador de la pinza en el modelo de Menagerie: 0 cerrada, 255 abierta

# Shadow Hand: dedos al frente e inclinados 35° hacia abajo, como al estirarse para agarrar algo de una mesa.
# Horizontal (0°) casi no se alcanza: la palma queda ~33 cm delante de la brida. Con 35° y esta caja, la IK
# quedó con 3 mm de error mediano y 9 mm como máximo. La postura preferida salió de una búsqueda con varias
# semillas para el centro de la caja.
SHADOW_TILT_DEG = 35
SHADOW_X_RANGE = (0.70, 0.50)
SHADOW_Y_RANGE = (0.25, -0.25)
SHADOW_Z_RANGE = (0.50, 0.25)
SHADOW_POSTURE = np.array([0.003, -0.841, -0.003, -2.583, -0.005, 2.701, 0.789])
# Doblez de tus dedos con la mano abierta (grados; medianas de las dos sesiones grabadas): es el cero del robot.
OPEN_DEG = {"FF": (20.5, 18.1, 35.1), "MF": (18.8, 27.1, 7.9), "RF": (17.2, 19.8, 10.5),
            "LF": (8.1, 27.9, 16.7), "TH": (10.2, 7.1, 12.8)}
DEAD_DEG = 5.0           # zona muerta sobre el cero, para que el ruido no mueva los dedos con la mano abierta
FINGER_GAIN = 1.2        # grados del robot por grado tuyo
THUMB_CM = (5.0, 2.5)    # pulgar-nudillo del índice: a 5 cm o más el pulgar abierto, a 2.5 cm o menos cerrado
FINGER_TAU_MS = 60       # suavizado de los dedos


def add_markers(spec, pos):
    """Esfera roja: objetivo filtrado. Esfera gris: la mano sin filtrar."""
    for name, size, rgba in (("target", 0.02, [1, 0, 0, 0.6]), ("raw", 0.012, [0.5, 0.5, 0.5, 0.5])):
        body = spec.worldbody.add_body(name=name, mocap=True, pos=pos)
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[size, 0, 0], rgba=rgba, contype=0, conaffinity=0)


def build_gripper_model():
    """Escena de Menagerie sin tocar sus archivos: se agregan la punta, la compensación de gravedad y los marcadores."""
    spec = mujoco.MjSpec.from_file(str(PANDA_SCENE))
    spec.body("hand").add_site(name="tcp", pos=[0, 0, TCP_OFFSET], size=[0.008, 0, 0], rgba=[1, 1, 0, 1])
    for name in GRIPPER_BODIES:
        spec.body(name).gravcomp = 1.0
    add_markers(spec, [0.55, 0, 0.52])
    return spec.compile()


def build_shadow_model():
    """Panda sin mano con la Shadow Hand montada en su brida, más cielo, piso y luz como el scene.xml de Menagerie."""
    panda = mujoco.MjSpec.from_file(str(PANDA_NOHAND))
    hand = mujoco.MjSpec.from_file(str(SHADOW))
    for mesh in hand.meshes:  # al unir, el meshdir del Panda no aplica a las mallas de la mano
        mesh.file = str(SHADOW.parent / "assets" / mesh.file)
    for spec in (panda, hand):
        for key in list(spec.keys):
            spec.delete(key)
    with warnings.catch_warnings():  # avisos de opciones distintas entre los dos modelos; se conservan las del Panda
        warnings.simplefilter("ignore")
        panda.site("attachment_site").attach_body(hand.body("rh_forearm"), "sh_", "").quat = [1, 0, 0, 0]
    panda.add_texture(name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX, builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                      rgb1=[0.3, 0.5, 0.7], rgb2=[0, 0, 0], width=512, height=3072)
    panda.add_texture(name="grid", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                      rgb1=[0.2, 0.3, 0.4], rgb2=[0.1, 0.2, 0.3], mark=mujoco.mjtMark.mjMARK_EDGE,
                      markrgb=[0.8, 0.8, 0.8], width=300, height=300)
    floor = panda.add_material(name="grid", texrepeat=[5, 5], texuniform=True, reflectance=0.2)
    floor.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "grid"
    panda.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05], material="grid")
    panda.worldbody.add_light(pos=[0, 0, 1.5], dir=[0, 0, -1], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL)
    panda.option.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT  # sin objetos que agarrar: sin contactos
    for body in panda.bodies:
        body.gravcomp = 1.0
    add_markers(panda, [0.6, 0, 0.4])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return panda.compile()


def tilted_down(deg):
    """Orientación de la palma: dedos (z) al frente e inclinados deg hacia abajo, palma (-y) hacia abajo."""
    t = math.radians(deg)
    fingers, back = np.array([math.cos(t), 0, -math.sin(t)]), np.array([math.sin(t), 0, math.cos(t)])
    return np.column_stack([np.cross(back, fingers), back, fingers])


def workspace_target(msg, x_range, y_range, z_range):
    dist_cm = armSim.DEPTH_K / msg["d"]
    return np.array([armSim.lin(dist_cm, 30, 65, *x_range),
                     armSim.lin(msg["u"], 0.20, 0.80, *y_range),
                     armSim.lin(msg["v"], 0.20, 0.80, *z_range)])


class PandaIK:
    """Cinemática inversa de 6 grados: mínimos cuadrados amortiguados y postura preferida en el espacio nulo."""

    def __init__(self, model, site, posture, R_des, damping=0.05, max_step=0.2, null_gain=0.3, iters=30):
        self.m, self.d = model, mujoco.MjData(model)
        self.site = model.site(site).id
        self.q_home, self.R_des = posture.copy(), R_des
        self.damping, self.max_step, self.null_gain, self.iters = damping, max_step, null_gain, iters
        self.lo, self.hi = model.jnt_range[:7, 0], model.jnt_range[:7, 1]
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


class Gripper:
    """Pinza abierta o cerrada según el pellizco, con histéresis y un mínimo de frames seguidos para cambiar."""

    def __init__(self, model):
        self.closed = False
        self.count = 0
        self.actuator = model.actuator("actuator8").id

    def update(self, data, msg):
        if "p" not in msg:  # un detector anterior no manda el pellizco: la pinza se queda como está
            return
        cm = msg["p"] * 100
        flip = cm > PINCH_OPEN_CM if self.closed else cm < PINCH_CLOSE_CM
        self.count = self.count + 1 if flip else 0
        if self.count >= PINCH_FRAMES:
            self.closed, self.count = not self.closed, 0
        data.ctrl[self.actuator] = 0.0 if self.closed else GRIPPER_OPEN


class ShadowFingers:
    """Dedos de la Shadow Hand desde el doblez de los tuyos, articulación a articulación."""

    def __init__(self, model):
        act = lambda name: model.actuator(f"sh_rh_A_{name}").id  # noqa: E731
        self.fingers = [(f, act(f"{f}J3"), act(f"{f}J0")) for f in ("FF", "MF", "RF", "LF")]
        self.thumb = {j: act(f"THJ{j}") for j in (1, 2, 4)}
        self.ids = [i for _, j3, j0 in self.fingers for i in (j3, j0)] + list(self.thumb.values())
        self.lo, self.hi = model.actuator_ctrlrange[self.ids].T
        self.smooth = filters.EMA(FINGER_TAU_MS)

    @staticmethod
    def bend(measured_rad, open_deg):
        return FINGER_GAIN * math.radians(max(0.0, math.degrees(measured_rad) - open_deg - DEAD_DEG))

    def update(self, data, msg):
        if "f" not in msg:  # un detector anterior no manda los dedos: la mano se queda como está
            return
        f = [msg["f"][i:i + 3] for i in range(0, 15, 3)]
        ctrl = []
        for k, (name, _, _) in enumerate(self.fingers):
            mcp, pip, dip = f[k]
            o = OPEN_DEG[name]
            ctrl += [self.bend(mcp, o[0]), self.bend(pip, o[1]) + self.bend(dip, o[2])]  # J0 = J2 + J1 (tendón)
        _, mcp, ip = f[4]
        o = OPEN_DEG["TH"]
        ctrl += [self.bend(ip, o[2]), self.bend(mcp, o[1]),
                 armSim.lin(msg["th"] * 100, *THUMB_CM, 0.0, self.hi[-1])]  # THJ1, THJ2, THJ4
        data.ctrl[self.ids] = np.clip(self.smooth.update(ctrl, msg["ts"] / 1000), self.lo, self.hi)


HANDS = {
    "gripper": dict(build=build_gripper_model, site="tcp", ranges=(X_RANGE, Y_RANGE, Z_RANGE), controller=Gripper,
                    camera=dict(azimuth=140, elevation=-20, distance=1.9, lookat=[0.45, 0.0, 0.35])),
    "shadow": dict(build=build_shadow_model, site="sh_grasp_site",
                   ranges=(SHADOW_X_RANGE, SHADOW_Y_RANGE, SHADOW_Z_RANGE), controller=ShadowFingers,
                   camera=dict(azimuth=150, elevation=-15, distance=1.6, lookat=[0.45, 0.0, 0.40])),
}


class PandaDemo:
    """Estado del demo. Separado del visor para poder probarlo sin ventana con una sesión grabada."""

    def __init__(self, filt, hand="shadow"):
        self.cfg = HANDS[hand]
        self.m = self.cfg["build"]()
        self.d = mujoco.MjData(self.m)
        if hand == "gripper":
            mujoco.mj_resetDataKeyframe(self.m, self.d, self.m.key("home").id)  # también pone ctrl, pinza abierta
            posture = self.m.key("home").qpos[:7].copy()
            mujoco.mj_kinematics(self.m, self.d)
            R_des = self.d.site_xmat[self.m.site("tcp").id].reshape(3, 3).copy()  # pinza hacia abajo, como en "home"
        else:
            posture, R_des = SHADOW_POSTURE, tilted_down(SHADOW_TILT_DEG)
            self.d.qpos[:7] = self.d.ctrl[:7] = posture
        mujoco.mj_forward(self.m, self.d)
        self.ik = PandaIK(self.m, self.cfg["site"], posture, R_des)
        self.hand = self.cfg["controller"](self.m)
        self.filt = filt
        self.q_des = posture.copy()
        self.target_id = self.m.body("target").mocapid[0]
        self.raw_id = self.m.body("raw").mocapid[0]

    def on_message(self, msg):
        raw = workspace_target(msg, *self.cfg["ranges"])
        target = self.filt.update(raw, msg["ts"] / 1000)
        self.q_des = self.ik.solve(target, self.q_des)
        self.d.ctrl[:7] = self.q_des
        self.hand.update(self.d, msg)
        self.d.mocap_pos[self.target_id] = target
        self.d.mocap_pos[self.raw_id] = raw
        return target

    def step_to(self, t):
        """Simula hasta el tiempo t (s de simulación)."""
        while self.d.time < t:
            mujoco.mj_step(self.m, self.d)


def main():
    parser = argparse.ArgumentParser(description="Demo del Franka Panda controlado por la mano. Correr con mjpython.")
    parser.add_argument("--hand", choices=HANDS, default="shadow")
    filters.add_arguments(parser)
    args = parser.parse_args()
    params = filters.params_from_arguments(args)
    demo = PandaDemo(filters.make_filter(args.filter, **params), args.hand)
    print(f"Mano: {args.hand} | filtro: {args.filter} {params}")

    sock = armSim.make_receiver()
    with mujoco.viewer.launch_passive(demo.m, demo.d) as viewer:
        cam = demo.cfg["camera"]
        viewer.cam.azimuth, viewer.cam.elevation, viewer.cam.distance = cam["azimuth"], cam["elevation"], cam["distance"]
        viewer.cam.lookat[:] = cam["lookat"]
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
