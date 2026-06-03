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
