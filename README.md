# Aerodynamic Coefficients Calculation

This is a Vilnius University Software Engineering master's thesis project. Its goal is to develop a software tool for calculating aerodynamic coefficients of aircraft to be used in Vehicle Dynamics Model based navigation.

The tool parses ArduPilot dataflash `.bin` logs and computes 6-DOF aerodynamic coefficients (CL, CD, CY, Cl, Cm, Cn) from IMU, attitude, and GPS data.

## Installation

Requires Python 3.12+.

```bash
pip install -e .
```

For development (pytest, ruff):

```bash
pip install -e ".[dev]"
```

## Usage

### 1. Parse a log file

Print a summary of message types and contents from an ArduPilot `.bin` log:

```bash
parse-log flight.bin
parse-log flight.bin -t ATT          # filter by message type
parse-log flight.bin -t GPS -n 50    # show first 50 GPS messages
parse-log flight.bin -a              # show all messages
```

### 2. Plot flight data

Visualize log data using built-in presets or custom field selection:

```bash
plot-log flight.bin att              # attitude (roll/pitch/yaw)
plot-log flight.bin gps              # 3D GPS trajectory
plot-log flight.bin imu              # IMU accelerometer & gyroscope
plot-log flight.bin baro             # barometric altitude
```

Custom field plotting:

```bash
plot-log flight.bin -t GPS -f Spd,Alt
plot-log flight.bin -t IMU -f AccX,AccY,AccZ --save plot.png
```

### 3. Compute aerodynamic coefficients

Create an aircraft configuration JSON file:

```json
{
    "mass": 1.5,
    "wing_area": 0.35,
    "wing_span": 1.4,
    "mean_aero_chord": 0.25,
    "i_xx": 0.029,
    "i_yy": 0.031,
    "i_zz": 0.055,
    "i_xz": 0.002
}
```

Then run:

```bash
calc-aero flight.bin aircraft.json
calc-aero flight.bin aircraft.json --plot            # show coefficient plots
calc-aero flight.bin aircraft.json --output results.csv  # save to CSV
calc-aero flight.bin aircraft.json --rho 1.1         # fixed air density
```

#### Aircraft config fields

| Field | Unit | Description |
|-------|------|-------------|
| `mass` | kg | Aircraft mass |
| `wing_area` | m² | Wing reference area (S) |
| `wing_span` | m | Wing span (b) |
| `mean_aero_chord` | m | Mean aerodynamic chord (c̄) |
| `i_xx`, `i_yy`, `i_zz` | kg·m² | Moments of inertia |
| `i_xz` | kg·m² | Product of inertia (default: 0) |

#### Output coefficients

| Coefficient | Description |
|-------------|-------------|
| CL | Lift coefficient |
| CD | Drag coefficient |
| CY | Side force coefficient |
| Cl | Rolling moment coefficient |
| Cm | Pitching moment coefficient |
| Cn | Yawing moment coefficient |

The output also includes angle of attack (alpha), sideslip angle (beta), dynamic pressure, and airspeed. Data points with dynamic pressure below 10 Pa (ground/taxi) are masked as NaN.

## Python API

```python
from acc.log_parser import parse_log
from acc.aero import compute_from_log
from acc.models import AircraftConfig

log_data = parse_log("flight.bin")

aircraft = AircraftConfig(
    mass=1.5, wing_area=0.35, wing_span=1.4,
    mean_aero_chord=0.25,
    i_xx=0.029, i_yy=0.031, i_zz=0.055, i_xz=0.002,
)

result = compute_from_log(log_data, aircraft)

# result.cl, result.cd, result.cy — force coefficients
# result.c_roll, result.cm, result.cn — moment coefficients
# result.alpha, result.beta, result.airspeed, result.dynamic_pressure
```

## Running tests

```bash
pytest
ruff check src/ tests/
```
