"""Environment package exports.

Keep imports lazy so lightweight modules such as `env.gate_detector` do not
pull in PyBullet when the competition-facing algorithm only needs perception.
"""

__all__ = [
    "AbstractDroneRacingEnv",
    "DEFAULT_GATES",
    "GateRaceAviary",
    "TRACK_LIBRARY",
    "get_track",
    "get_tracks",
    "make_env",
]


def __getattr__(name):
    if name == "AbstractDroneRacingEnv":
        from env.abstract_env import AbstractDroneRacingEnv

        return AbstractDroneRacingEnv
    if name in {"GateRaceAviary", "DEFAULT_GATES", "make_env"}:
        from env.gate_race_aviary import DEFAULT_GATES, GateRaceAviary, make_env

        return {
            "GateRaceAviary": GateRaceAviary,
            "DEFAULT_GATES": DEFAULT_GATES,
            "make_env": make_env,
        }[name]
    if name in {"TRACK_LIBRARY", "get_track", "get_tracks"}:
        from env.tracks import TRACK_LIBRARY, get_track, get_tracks

        return {
            "TRACK_LIBRARY": TRACK_LIBRARY,
            "get_track": get_track,
            "get_tracks": get_tracks,
        }[name]
    raise AttributeError(f"module 'env' has no attribute {name!r}")
