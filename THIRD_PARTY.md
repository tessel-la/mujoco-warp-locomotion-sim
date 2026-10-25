# Third-party components

The root MIT license covers tessel·la's integration code. It does not relicense the
following components or their generated derivatives.

| Component | Source and terms | Local notice |
| --- | --- | --- |
| Microduck CAD/MJCF and meshes, including training-tree copies | Pollen Robotics `microduck_rl`, revision `c690e5599866893b11b3ab2764883a8b0800cbe7`; upstream states Creative Commons BY-SA-NC for 3D model files | [Asset source](assets/microduck/SOURCE.md), [preserved upstream statement](assets/microduck/MODEL-LICENSE.md) |
| Vendored Microduck RL code | Same upstream revision; Apache-2.0 | [Training license](training/microduck_rl/LICENSE), [upstream provenance](training/microduck_rl/UPSTREAM.md) |
| Seven deployment ONNX policies | Pollen Robotics' Hugging Face simulator, immutable revision and digests in `policies/manifest.json`; Apache-2.0 | [Policy source](policies/SOURCE.md), [license](policies/LICENSE) |
| Open Duck Mini v2 reference assets | Upstream revision and modifications documented locally; Apache-2.0 | [Source](assets/duckbot/open_duck_mini_v2/SOURCE.md), [license](assets/duckbot/open_duck_mini_v2/LICENSE) |

The generated Microduck visualization URDF derives from the restricted model; it is
not a way to remove those terms. The pinned upstream README does not identify the
Creative Commons license version or provide its complete model license text. The
original statement is preserved verbatim instead of guessing a version. Obtain the
model owner's clarification or permission for commercial use or uncertain redistribution.

Runtime libraries such as MuJoCo, MuJoCo Warp, ROS, and ONNX Runtime remain separate
dependencies with their upstream licenses and notices. Training has its own locked
runtime; it is not installed by the operator or mock image.
