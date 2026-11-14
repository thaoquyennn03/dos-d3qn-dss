"""Shared constants. Units follow the user's source dataset: km, minutes, CNY."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = Path(os.environ.get("DOS_RESULTS", ROOT / "results"))   # override with env var DOS_RESULTS

GRID_KM = 10.0            # 10 x 10 km Manhattan grid (source data)
SHIFT_MIN = 480           # 8-hour shift
SPEED_KM_MIN = 28 / 60    # 28 km/h (assumed_speed_kmh in source batch file)
COST_PER_KM = 0.9         # CNY / km  (source README)
FEE_BASE, FEE_PER_KM = 4.0, 0.5   # fee = 4 + 0.5 * restaurant->customer km (source README)
TIP_GLOBAL_MEAN, TIP_SD = 4.5, 2.0

ASSIGN_EVERY = 2          # platform assigns every 2 min (paper Table 2)
REJECT_LOCK = 10          # rejected order not re-offered to same courier for 10 min (paper Sec 3.1)
ORDER_EXPIRE = 45         # pending order is lost after 45 min
MAX_PICKUP_KM = 8.0       # platform only offers to couriers within 8 km of the restaurant
COMP_ACCEPT = 0.8         # competitors accept 80% of offers (paper static policies)
TAU = 5                   # idle repositioning decision every 5 min
HOURLY_WEIGHTS = [2.5, 3, 4, 7, 6, 4, 3, 2]   # source README

# Demand levels (orders per day for the whole area) with N_COMPETITORS couriers.
N_COMPETITORS = 30
LEVELS = {"few": 150, "sufficient": 400, "excess": 1200}
PATTERNS = ["stationary", "peak_fixed", "hotspot_random"]
HOT_PER_BLOCK, HOT_MULT = 6, 4.0   # 6 of 30 restaurants x4 demand in each 2-h block

# DOS settings
DOS_WINDOW = 30           # minutes of recent orders
DOS_RADIUS = 2.5          # km (Manhattan)
NEIGHBOR_STEP = 1.5       # km offset for the 4 neighbour cells

ACTIONS = ["accept", "reject", "stay", "up", "down", "left", "right"]
OFFER_ACTIONS = [0, 1]
MOVE_ACTIONS = [2, 3, 4, 5, 6]
DIRS = {2: (0, 0), 3: (0, 1), 4: (0, -1), 5: (-1, 0), 6: (1, 0)}
