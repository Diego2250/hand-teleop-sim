# Third-party models

## franka_emika_panda

Unmodified copy of the Franka Emika Panda description from
[MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/main/franka_emika_panda),
commit `c96a32d28fb5da84da38c1da4d749e7a13212855` (2026-09-23), by Google DeepMind, derived
from Franka Emika's public URDF. Licensed under the Apache License 2.0 (see
`franka_emika_panda/LICENSE`).

`pandaSim.py` loads `franka_emika_panda/scene.xml` and adds its own sites and target markers at
load time, so the files in this folder are kept exactly as upstream.
