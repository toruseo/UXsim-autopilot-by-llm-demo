"""
Generate synthetic traffic data using UXsim.

Creates a 5x5 grid network with time-varying demand that produces
significant congestion (without gridlock) for travel time prediction.
A Gaussian peak demand pattern creates realistic congestion buildup
and dissipation.
"""

import os
import random
import itertools

import numpy as np
import pandas as pd
from uxsim import World

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
NUM_SCENARIOS = 30          # total simulation scenarios
GRID_SIZE = 5               # 5x5 grid
LINK_LENGTH = 500           # metres
FREE_FLOW_SPEED = 50 / 3.6  # km/h -> m/s  (~13.9 m/s)
JAM_DENSITY = 0.2           # veh/m
SIM_DURATION = 7200          # seconds (2 hours, for congestion cycle)
DEMAND_INTERVAL = 300        # seconds – resolution for demand slicing
PEAK_FACTOR = 2.0            # peak-to-base demand ratio
RECORD_DT = 30               # seconds – sampling interval for records
OUTPUT_DIR = "data"

np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)


def build_network(W):
    """Create a 5x5 grid network and return a dict of node references."""
    nodes = {}
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            name = f"N{i}_{j}"
            nodes[(i, j)] = W.addNode(name, i * LINK_LENGTH, j * LINK_LENGTH)

    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            if j + 1 < GRID_SIZE:
                W.addLink(f"L{i}_{j}_to_{i}_{j+1}",
                          nodes[(i, j)], nodes[(i, j + 1)],
                          length=LINK_LENGTH,
                          free_flow_speed=FREE_FLOW_SPEED,
                          jam_density=JAM_DENSITY)
                W.addLink(f"L{i}_{j+1}_to_{i}_{j}",
                          nodes[(i, j + 1)], nodes[(i, j)],
                          length=LINK_LENGTH,
                          free_flow_speed=FREE_FLOW_SPEED,
                          jam_density=JAM_DENSITY)
            if i + 1 < GRID_SIZE:
                W.addLink(f"L{i}_{j}_to_{i+1}_{j}",
                          nodes[(i, j)], nodes[(i + 1, j)],
                          length=LINK_LENGTH,
                          free_flow_speed=FREE_FLOW_SPEED,
                          jam_density=JAM_DENSITY)
                W.addLink(f"L{i+1}_{j}_to_{i}_{j}",
                          nodes[(i + 1, j)], nodes[(i, j)],
                          length=LINK_LENGTH,
                          free_flow_speed=FREE_FLOW_SPEED,
                          jam_density=JAM_DENSITY)
    return nodes


def add_demand(W, nodes, demand_scale):
    """
    Add OD demand between boundary nodes with time-varying pattern.

    A Gaussian peak centred at 40% of the simulation creates a realistic
    congestion build-up and dissipation cycle.  ``demand_scale`` controls
    overall volume (higher -> more congestion).
    """
    boundary = []
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            if i == 0 or i == GRID_SIZE - 1 or j == 0 or j == GRID_SIZE - 1:
                boundary.append((i, j))

    od_pairs = [(o, d) for o, d in itertools.product(boundary, boundary)
                if o != d and abs(o[0] - d[0]) + abs(o[1] - d[1]) >= 3]

    peak_centre = SIM_DURATION * 0.4
    peak_sigma = SIM_DURATION * 0.15

    for o, d in od_pairs:
        base_rate = demand_scale * np.random.uniform(0.015, 0.04)
        for t_start in range(0, SIM_DURATION, DEMAND_INTERVAL):
            t_mid = t_start + DEMAND_INTERVAL / 2
            time_factor = 1.0 + (PEAK_FACTOR - 1.0) * np.exp(
                -((t_mid - peak_centre) ** 2) / (2 * peak_sigma ** 2))
            rate = base_rate * time_factor * np.random.uniform(0.7, 1.3)
            W.adddemand(nodes[o], nodes[d], t_start,
                        min(t_start + DEMAND_INTERVAL, SIM_DURATION),
                        flow=rate)


def run_scenario(scenario_id, demand_scale):
    """Run one simulation scenario and return link-level records."""
    W = World(name=f"scenario_{scenario_id}",
              deltan=5,
              tmax=SIM_DURATION,
              print_mode=0,
              save_mode=0)

    nodes = build_network(W)
    add_demand(W, nodes, demand_scale)

    W.exec_simulation()
    W.analyzer.basic_analysis()

    dt = W.DELTAT  # simulation time-step (seconds)
    records = []

    for link in W.LINKS:
        tt_instant = link.traveltime_instant  # length = TMAX / DELTAT
        cum_arr = link.cum_arrival
        cum_dep = link.cum_departure
        free_flow_tt = link.length / link.free_flow_speed
        n_steps = len(tt_instant)

        sample_step = max(1, int(RECORD_DT / dt))
        for step in range(0, n_steps, sample_step):
            t = step * dt
            tt = float(tt_instant[step])
            # number of vehicles = cumulative arrivals - departures
            n_veh = (cum_arr[step] - cum_dep[step]) if step < len(cum_arr) else 0
            density_est = n_veh / link.length if link.length > 0 else 0

            records.append({
                "scenario": scenario_id,
                "time": t,
                "link": link.name,
                "length": link.length,
                "free_flow_speed": link.free_flow_speed,
                "free_flow_tt": free_flow_tt,
                "num_vehicles": n_veh,
                "density": density_est,
                "travel_time": tt,
                "demand_scale": demand_scale,
            })

    return records


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_records = []
    demand_scales = np.random.uniform(0.7, 1.0, size=NUM_SCENARIOS)

    for sid in range(NUM_SCENARIOS):
        print(f"Running scenario {sid + 1}/{NUM_SCENARIOS} "
              f"(demand_scale={demand_scales[sid]:.3f}) ...")
        recs = run_scenario(sid, demand_scales[sid])
        all_records.extend(recs)

    df = pd.DataFrame(all_records)

    # Remove unrealistic rows where travel_time is extremely large
    tt_cap = df["free_flow_tt"] * 5
    df = df[df["travel_time"] <= tt_cap]

    out_path = os.path.join(OUTPUT_DIR, "traffic_data.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved {len(df)} records to {out_path}")

    ff_tt = df["free_flow_tt"].iloc[0]
    print(f"\nCongestion statistics:")
    print(f"  At free-flow (TT = {ff_tt:.0f}s): "
          f"{(df['travel_time'] <= ff_tt * 1.01).mean()*100:.1f}%")
    print(f"  Congested (TT > 1.1x free-flow): "
          f"{(df['travel_time'] > ff_tt * 1.1).mean()*100:.1f}%")
    print(f"  Heavily congested (TT > 1.5x free-flow): "
          f"{(df['travel_time'] > ff_tt * 1.5).mean()*100:.1f}%")
    print(df["travel_time"].describe())


if __name__ == "__main__":
    main()
