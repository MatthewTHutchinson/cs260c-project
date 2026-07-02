# External Harness Patches

This folder preserves local changes made to sibling third-party repositories.

## `elodin-ai-grand-prix-cs260c.patch`

Patch for:

```text
/Users/matthewhutchinson/dev/elodin-ai-grand-prix
```

Why it exists:

- the Elodin checkout's Git remote points to upstream `elodin-sys/ai-grand-prix`
- the CS260C course repo still needs a reproducible record of local harness changes
- the patch includes the inline Betaflight smoke-test support, FPV camera
  profile experiments, custom courses, and `solver/cs260c_pilot.py`
- `solver/cs260c_pilot.py` selects the active detector through the CS260C
  algorithm package, so `CS260C_GATE_DETECTOR=gatenet` can be tested without
  editing the sibling solver

Apply from the Elodin repo root with:

```bash
git apply /Users/matthewhutchinson/dev/cs260c-project/patches/elodin-ai-grand-prix-cs260c.patch
```

The Betaflight submodule may still show local build changes after `scripts/build_betaflight.sh`; those are build artifacts and are not part of this patch.

## `gym-pybullet-drones-local.patch.gz`

Final local changes from the standalone checkout that was removed during
project archival. The patch was generated against upstream commit:

```text
82b0fda6d005b287cdc1ef0c313d09a361027c45
```

Restore with:

```bash
git clone https://github.com/utiasDSL/gym-pybullet-drones.git
cd gym-pybullet-drones
git checkout 82b0fda6d005b287cdc1ef0c313d09a361027c45
gzip -dc ../cs260c-project/patches/gym-pybullet-drones-local.patch.gz | git apply -
```

This checkout is historical and is not the active project direction.
