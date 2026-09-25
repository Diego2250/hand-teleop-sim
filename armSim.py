import math
import time
import numpy as np
import mujoco
import mujoco.viewer
import json
import socket

UDP_ADDR = ("127.0.0.1", 5005)
SHOULDER = np.array([0.0, 0.0, 0.15])
MAX_REACH = 0.62  # 95% del alcance (0.35 + 0.30 m): evita el brazo totalmente estirado (singular)
DEPTH_K = 81.2    # distancia mano-cámara en cm ≈ DEPTH_K / d; calibrado con tools/depth_probe.py (cámara 1920x1080)
HOME_TARGET = np.array([0.3, 0.0, 0.45])  # objetivo antes del primer mensaje

ARM_XML = """
<mujoco>
  <option timestep="0.002" gravity="0 0 0"/>
  <default><joint damping="2"/></default>
  <worldbody>
    <light pos="0 0 3"/>
    <geom type="plane" size="2 2 0.1" rgba="0.8 0.8 0.8 1"/>
    <body name="base" pos="0 0 0.1">
      <joint name="yaw" type="hinge" axis="0 0 1"/>
      <geom type="cylinder" size="0.05 0.05" rgba="0.3 0.3 0.3 1"/>
      <body name="upper" pos="0 0 0.05">
        <joint name="shoulder" type="hinge" axis="0 1 0"/>
        <geom type="capsule" fromto="0 0 0 0 0 0.35" size="0.03" rgba="0.2 0.5 0.9 1"/>
        <body name="fore" pos="0 0 0.35">
          <joint name="elbow" type="hinge" axis="0 1 0"/>
          <geom type="capsule" fromto="0 0 0 0 0 0.3" size="0.025" rgba="0.2 0.7 0.4 1"/>
          <site name="ee" pos="0 0 0.3" size="0.015" rgba="1 1 0 1"/>
        </body>
      </body>
    </body>
    <body name="target" mocap="true" pos="0.3 0 0.45">
      <geom type="sphere" size="0.03" rgba="1 0 0 0.6" contype="0" conaffinity="0"/>
    </body>
  </worldbody>
  <actuator>
    <position joint="yaw" kp="200"/>
    <position joint="shoulder" kp="200"/>
    <position joint="elbow" kp="200"/>
  </actuator>
</mujoco>
"""

def solve_ik(model, ik_data, site_id, target, q_init, iters=20, damping=1e-2, max_step=0.2):
    ik_data.qpos[:] = q_init
    jacp = np.zeros((3, model.nv))
    for _ in range(iters):
        mujoco.mj_fwdPosition(model, ik_data)
        err = target - ik_data.site_xpos[site_id]
        if np.linalg.norm(err) < 1e-4:
            break
        mujoco.mj_jacSite(model, ik_data, jacp, None, site_id)
        dq = jacp.T @ np.linalg.solve(jacp @ jacp.T + damping**2 * np.eye(3), err)
        n = np.linalg.norm(dq)
        if n > max_step:  # cerca de una singularidad dq explota; se limita el paso
            dq *= max_step / n
        ik_data.qpos[:] += dq
    return ik_data.qpos.copy()

def make_sim():
    model = mujoco.MjModel.from_xml_string(ARM_XML)
    data = mujoco.MjData(model)
    ik_data = mujoco.MjData(model)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee")
    return model, data, ik_data, site_id

def control_step(model, data, ik_data, site_id, target):
    """Un paso de simulación persiguiendo target. Lo usan el loop en vivo y tools/replay.py."""
    data.mocap_pos[0] = target
    data.ctrl[:] = solve_ik(model, ik_data, site_id, target, data.qpos)
    mujoco.mj_step(model, data)

def main():
    model, data, ik_data, site_id = make_sim()

    sock = make_receiver()
    target = HOME_TARGET.copy()
    step = 0
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            msg = read_latest(sock)
            if msg is not None:
                target = clamp_reach(hand_to_target(msg))

            control_step(model, data, ik_data, site_id, target)

            if step % 500 == 0:
                err = np.linalg.norm(data.site_xpos[site_id] - target) * 1000
                print(f"error de seguimiento: {err:.1f} mm")

            viewer.sync()
            time.sleep(model.opt.timestep)
            step += 1

def make_receiver():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(UDP_ADDR)
    sock.setblocking(False)
    return sock

def read_latest(sock):
    msg = None
    while True:
        try:
            data, _ = sock.recvfrom(1024)
            msg = json.loads(data)
        except BlockingIOError:
            return msg

def lin(val, in_lo, in_hi, out_lo, out_hi):
    t = min(max((val - in_lo) / (in_hi - in_lo), 0.0), 1.0)
    return out_lo + t * (out_hi - out_lo)

def hand_to_target(msg):
    dist_cm = DEPTH_K / msg["d"]
    x = lin(dist_cm, 30, 65, 0.40, 0.15)        # profundidad: mano cerca de la cámara, brazo adelante
    y = lin(msg["u"], 0.20, 0.80, 0.25, -0.25)  # izquierda y derecha
    z = lin(msg["v"], 0.20, 0.80, 0.60, 0.25)   # arriba y abajo (v crece hacia abajo en la imagen)
    return np.array([x, y, z])

def clamp_reach(p):
    d = p - SHOULDER
    n = np.linalg.norm(d)
    return SHOULDER + d * (MAX_REACH / n) if n > MAX_REACH else p

if __name__ == "__main__":
    main()