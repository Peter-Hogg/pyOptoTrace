# Third-party and provenance review

pyOptoTrace depends on third-party Python packages, but does not vendor them. Their licenses apply
when installed and should be reviewed before redistribution in a bundled application.

Two source modules also contain explicit method/provenance notes inherited from the research
repository:

- `kymograph.py` describes an independent Python implementation based on Cohen Lab MATLAB
  kymograph workflows (`polyLineKymo2.m`, `apply_polyLineKymo2.m`, `lineKymo.m`, and
  `wedgeKymo.m`). Confirm the original code's license and the degree of derivation before release.
- `spike_tools.py` describes `find_spike_bh` as a Python translation of a MATLAB method. Confirm
  permission, add the canonical paper/code citation, or replace it with a clean-room
  implementation before release.

Comments elsewhere mention published or external methods such as NoRMCorre, PMD/localmd, NWB,
Pynapple, and PyWavelets. Those projects are not copied into this repository. Add precise paper
citations to the documentation as the scientific API is stabilized.
