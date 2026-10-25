# Microduck deployment policies

The seven ONNX files named in `manifest.json` come from Pollen Robotics' public
[Microduck simulator](https://huggingface.co/spaces/pollen-robotics/microduck-simulator/tree/ad9d41a763f30dc770817c33d92dcfec0205db1e/app/public/policies),
revision `ad9d41a763f30dc770817c33d92dcfec0205db1e`. The space declares Apache-2.0;
its license is preserved in [LICENSE](LICENSE). These policy terms do not relicense
the separately attributed robot CAD models.

Run `python tools/download_policies.py` from the repository root. It downloads only
missing or mismatched files from that revision, verifies every SHA-256 digest in
the manifest, and atomically replaces each verified file. A failed download leaves
any existing file intact. No Hugging Face account or token is required.
