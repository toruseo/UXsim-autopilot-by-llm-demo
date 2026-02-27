"""
Generate synthetic traffic data using UXsim.

Creates a set of parallel corridor networks, each with a capacity
bottleneck that naturally produces sustained congestion upstream.
This design concentrates traffic flow (no alternative routes) so that
a large fraction of recorded travel times reflect congested conditions.
"""

import os
import random

import numpy as np
import pandas as pd
from uxsim import World

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
NUM_SCENARIOS = 30               # total simulation scenarios
NUM_CORRIDORS = 10               # parallel corridors per scenario
NODES_PER_CORRIDOR = 8           # nodes per corridor (7 links each)
LINK_LENGTH = 300                # metres
FREE_FLOW_SPEED = 50 / 3.6      # km/h -> m/s  (~13.9 m/s)
JAM_DENSITY = 0.2                # veh/m  (normal links)
BOTTLENECK_JAM_DENSITY = 0.05    # veh/m  (bottleneck links — 25% capacity)
SIM_DURATION = 3600              # seconds (1 hour)
DEMAND_INTERVAL = 300            # seconds – resolution for demand slicing
DEMAND_SCALE_MIN = 1.5           # minimum demand scale across scenarios
DEMAND_SCALE_MAX = 3.0           # maximum demand scale across scenarios
RECORD_DT = 30                   # seconds – sampling interval for records
OUTPUT_DIR = "data"

np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)


def build_network(W):
    """Create parallel corridor networks with bottleneck links.

    Each corridor is a linear sequence of nodes connected by
    one-directional links.  One link per corridor has reduced
    capacity (bottleneck) placed near the downstream end so that
    queues build upstream through most of the corridor.

    Returns
    -------
    nodes : dict
        Mapping (corridor, position) -> node reference.
    btl_positions : ndarray
        Bottleneck link position for each corridor.
    """
    nodes = {}
    for c in range(NUM_CORRIDORS):
        for p in range(NODES_PER_CORRIDOR):
            name = f"N{c}_{p}"
            nodes[(c, p)] = W.addNode(
                name, p * LINK_LENGTH, c * LINK_LENGTH * 3)

    # Place bottleneck in the downstream half of each corridor so that
    # queues propagate upstream through most of the corridor's links.
    btl_positions = np.random.randint(
        NODES_PER_CORRIDOR - 4, NODES_PER_CORRIDOR - 1, size=NUM_CORRIDORS)

    for c in range(NUM_CORRIDORS):
        for p in range(NODES_PER_CORRIDOR - 1):
            is_btl = (p == btl_positions[c])
            jd = BOTTLENECK_JAM_DENSITY if is_btl else JAM_DENSITY
            W.addLink(f"L{c}_{p}",
                      nodes[(c, p)], nodes[(c, p + 1)],
                      length=LINK_LENGTH,
                      free_flow_speed=FREE_FLOW_SPEED,
                      jam_density=jd)

    return nodes, btl_positions


def add_demand(W, nodes, demand_scale):
    """Add sustained demand along each corridor.

    A trapezoidal time profile provides a short ramp-up, a long
    plateau of high demand (exceeding bottleneck capacity), and a
    brief wind-down.  This keeps links upstream of the bottleneck
    congested for most of the simulation.
    """
    for c in range(NUM_CORRIDORS):
        base_rate = demand_scale * np.random.uniform(0.15, 0.35)
        for t_start in range(0, SIM_DURATION, DEMAND_INTERVAL):
            t_mid = t_start + DEMAND_INTERVAL / 2
            # Trapezoidal profile: ramp 0-10%, plateau 10-85%, wind-down 85-100%
            if t_mid < SIM_DURATION * 0.1:
                time_factor = t_mid / (SIM_DURATION * 0.1)
            elif t_mid < SIM_DURATION * 0.85:
                time_factor = 1.0
            else:
                # Keep a small residual flow to avoid empty-network artefacts
                time_factor = max(0.05, 1.0 - (t_mid - SIM_DURATION * 0.85)
                                  / (SIM_DURATION * 0.15))
            rate = base_rate * time_factor * np.random.uniform(0.8, 1.2)
            W.adddemand(nodes[(c, 0)],
                        nodes[(c, NODES_PER_CORRIDOR - 1)],
                        t_start,
                        min(t_start + DEMAND_INTERVAL, SIM_DURATION),
                        flow=rate)


def run_scenario(scenario_id, demand_scale):
    """Run one simulation scenario and return link-level records."""
    W = World(name=f"scenario_{scenario_id}",
              deltan=5,
              tmax=SIM_DURATION,
              print_mode=0,
              save_mode=0)

    nodes, btl_positions = build_network(W)
    add_demand(W, nodes, demand_scale)

    W.exec_simulation()
    W.analyzer.basic_analysis()

    dt = W.DELTAT
    records = []

    for link in W.LINKS:
        tt_instant = link.traveltime_instant
        cum_arr = link.cum_arrival
        cum_dep = link.cum_departure
        free_flow_tt = link.length / link.free_flow_speed
        n_steps = len(tt_instant)

        sample_step = max(1, int(RECORD_DT / dt))
        for step in range(0, n_steps, sample_step):
            t = step * dt
            tt = float(tt_instant[step])
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
    demand_scales = np.random.uniform(DEMAND_SCALE_MIN, DEMAND_SCALE_MAX,
                                      size=NUM_SCENARIOS)

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

    # All links share the same length and speed in this network.
    ff_tt = df["free_flow_tt"].iloc[0]
    print(f"\nCongestion statistics:")
    print(f"  At free-flow (TT = {ff_tt:.1f}s): "
          f"{(df['travel_time'] <= ff_tt * 1.01).mean()*100:.1f}%")
    print(f"  Congested (TT > 1.1x free-flow): "
          f"{(df['travel_time'] > ff_tt * 1.1).mean()*100:.1f}%")
    print(f"  Heavily congested (TT > 1.5x free-flow): "
          f"{(df['travel_time'] > ff_tt * 1.5).mean()*100:.1f}%")
    print(df["travel_time"].describe())


if __name__ == "__main__":
    main()
