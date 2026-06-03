"""Competition-facing autonomous racing algorithm components."""

from algorithm.autopilot import AutonomousRacingPilot
from algorithm.camera_profiles import CameraProfile, get_camera_profile
from algorithm.detector_factory import (
    DetectorFactoryConfig,
    build_gate_detector,
    build_gate_tracker,
    detector_config_from_env,
)
from algorithm.gate_detector import CameraParams, GateDetector, GateObservation
from algorithm.gate_tracker import GateTracker
from algorithm.neural_gate_detector import (
    GateNetONNXDetector,
    NeuralGateDetectorConfig,
    OpenCVDNNGateDetector,
)
from algorithm.types import GateEstimate, RacingCommand, TrackMode, VehicleTelemetry

__all__ = [
    "AutonomousRacingPilot",
    "CameraParams",
    "CameraProfile",
    "DetectorFactoryConfig",
    "GateNetONNXDetector",
    "GateDetector",
    "GateObservation",
    "GateEstimate",
    "GateTracker",
    "NeuralGateDetectorConfig",
    "OpenCVDNNGateDetector",
    "RacingCommand",
    "TrackMode",
    "VehicleTelemetry",
    "build_gate_detector",
    "build_gate_tracker",
    "detector_config_from_env",
    "get_camera_profile",
]
