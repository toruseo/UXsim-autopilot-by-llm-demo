"""
Generate traffic simulation visualizations using UXsim's built-in features.

Runs one representative scenario from the interconnected corridor network
and produces:
  - Network state animation (GIF)
  - Network state snapshots at key time points (before / during / after congestion)
  - Time-space trajectory diagram for a single corridor
  - Time-space density diagram for a single corridor
  - Cumulative arrival/departure curves for a single corridor
  - Macroscopic fundamental diagram
"""

import os
import shutil
import random

import numpy as np
import matplotlib
matplotlib.use("Agg")
from uxsim import World

# ---------------------------------------------------------------------------
# Configuration (matches generate_data.py)
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
NUM_CORRIDORS = 10
NODES_PER_CORRIDOR = 8
LINK_LENGTH = 300                # metres
FREE_FLOW_SPEED = 50 / 3.6      # km/h -> m/s
JAM_DENSITY = 0.2                # veh/m  (normal links)
BOTTLENECK_JAM_DENSITY = 0.05    # veh/m  (bottleneck links)
CROSS_LINK_POSITIONS = [2, 4, 6] # node positions where corridors connect
CROSS_LINK_LENGTH = 300          # metres  (lateral links)
CROSS_DEMAND_FRACTION = 0.3      # fraction of base rate for cross-corridor OD
SIM_DURATION = 3600              # seconds (1 hour)
DEMAND_INTERVAL = 300            # seconds
DEMAND_SCALE = 2.5               # representative mid-range demand
OUTPUT_DIR = "results"

np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)


def build_network(W):
    """Create interconnected corridor network with bottleneck links."""
    nodes = {}
    for c in range(NUM_CORRIDORS):
        for p in range(NODES_PER_CORRIDOR):
            name = f"N{c}_{p}"
            nodes[(c, p)] = W.addNode(
                name, c * LINK_LENGTH * 3, p * LINK_LENGTH)

    btl_positions = np.random.randint(
        NODES_PER_CORRIDOR - 4, NODES_PER_CORRIDOR - 1, size=NUM_CORRIDORS)

    links_by_corridor = {}
    for c in range(NUM_CORRIDORS):
        corridor_links = []
        for p in range(NODES_PER_CORRIDOR - 1):
            is_btl = (p == btl_positions[c])
            jd = BOTTLENECK_JAM_DENSITY if is_btl else JAM_DENSITY
            link = W.addLink(f"L{c}_{p}",
                             nodes[(c, p)], nodes[(c, p + 1)],
                             length=LINK_LENGTH,
                             free_flow_speed=FREE_FLOW_SPEED,
                             jam_density=jd)
            corridor_links.append(link)
        links_by_corridor[c] = corridor_links

    # Lateral cross-links between adjacent corridors
    for c in range(NUM_CORRIDORS - 1):
        for p in CROSS_LINK_POSITIONS:
            W.addLink(f"X{c}to{c+1}_{p}",
                      nodes[(c, p)], nodes[(c + 1, p)],
                      length=CROSS_LINK_LENGTH,
                      free_flow_speed=FREE_FLOW_SPEED,
                      jam_density=JAM_DENSITY)
            W.addLink(f"X{c+1}to{c}_{p}",
                      nodes[(c + 1, p)], nodes[(c, p)],
                      length=CROSS_LINK_LENGTH,
                      free_flow_speed=FREE_FLOW_SPEED,
                      jam_density=JAM_DENSITY)

    return nodes, btl_positions, links_by_corridor


def add_demand(W, nodes, demand_scale):
    """Add sustained demand along each corridor and between adjacent corridors."""
    for c in range(NUM_CORRIDORS):
        base_rate = demand_scale * np.random.uniform(0.15, 0.35)
        for t_start in range(0, SIM_DURATION, DEMAND_INTERVAL):
            t_mid = t_start + DEMAND_INTERVAL / 2
            if t_mid < SIM_DURATION * 0.1:
                time_factor = t_mid / (SIM_DURATION * 0.1)
            elif t_mid < SIM_DURATION * 0.85:
                time_factor = 1.0
            else:
                time_factor = max(0.05, 1.0 - (t_mid - SIM_DURATION * 0.85)
                                  / (SIM_DURATION * 0.15))
            rate = base_rate * time_factor * np.random.uniform(0.8, 1.2)
            W.adddemand(nodes[(c, 0)],
                        nodes[(c, NODES_PER_CORRIDOR - 1)],
                        t_start,
                        min(t_start + DEMAND_INTERVAL, SIM_DURATION),
                        flow=rate)

    # Cross-corridor demand between adjacent corridors
    for c in range(NUM_CORRIDORS - 1):
        cross_rate = demand_scale * np.random.uniform(0.05, 0.15)
        for t_start in range(0, SIM_DURATION, DEMAND_INTERVAL):
            t_mid = t_start + DEMAND_INTERVAL / 2
            if t_mid < SIM_DURATION * 0.1:
                time_factor = t_mid / (SIM_DURATION * 0.1)
            elif t_mid < SIM_DURATION * 0.85:
                time_factor = 1.0
            else:
                time_factor = max(0.05, 1.0 - (t_mid - SIM_DURATION * 0.85)
                                  / (SIM_DURATION * 0.15))
            rate = cross_rate * time_factor * np.random.uniform(0.8, 1.2)
            t_end = min(t_start + DEMAND_INTERVAL, SIM_DURATION)
            W.adddemand(nodes[(c, 0)],
                        nodes[(c + 1, NODES_PER_CORRIDOR - 1)],
                        t_start, t_end, flow=rate * CROSS_DEMAND_FRACTION)
            W.adddemand(nodes[(c + 1, 0)],
                        nodes[(c, NODES_PER_CORRIDOR - 1)],
                        t_start, t_end, flow=rate * CROSS_DEMAND_FRACTION)


def copy_uxsim_output(sim_output_dir, dest_dir, src_name, dest_name):
    """Copy a file from UXsim's output directory to the results directory."""
    src = os.path.join(sim_output_dir, src_name)
    dst = os.path.join(dest_dir, dest_name)
    if os.path.exists(src):
        shutil.copy2(src, dst)
        print(f"  Saved: {dst}")
    else:
        print(f"  WARNING: {src} not found")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Running representative scenario for visualizations ...")
    W = World(name="vis",
              deltan=5,
              tmax=SIM_DURATION,
              print_mode=0,
              save_mode=1)

    nodes, btl_positions, links_by_corridor = build_network(W)
    add_demand(W, nodes, DEMAND_SCALE)

    W.exec_simulation()
    W.analyzer.basic_analysis()

    sim_outdir = f"out{W.name}"
    print(f"  Completed trips: {W.analyzer.trip_completed} / {W.analyzer.trip_all}")
    print(f"  Average travel time: {W.analyzer.average_travel_time:.1f} s")

    # ------------------------------------------------------------------
    # 1. Network state animation (GIF)
    # ------------------------------------------------------------------
    print("Generating network animation ...")
    anim_path = os.path.join(OUTPUT_DIR, "network_animation.gif")
    W.analyzer.network_anim(
        file_name=anim_path,
        animation_speed_inverse=10,
        detailed=0,
        figsize=(16, 5),
        timestep_skip=8,
        network_font_size=0,
        node_size=4,
    )
    print(f"  Saved: {anim_path}")

    # ------------------------------------------------------------------
    # 2. Network state snapshots at key times
    #    UXsim saves these to out<name>/network<detailed>_<t>.png
    # ------------------------------------------------------------------
    snapshot_times = {
        "early": int(SIM_DURATION * 0.1),
        "peak": int(SIM_DURATION * 0.5),
        "late": int(SIM_DURATION * 0.9),
    }
    print("Generating network snapshots ...")
    for label, t in snapshot_times.items():
        W.analyzer.network(
            t=t,
            detailed=1,
            figsize=(16, 5),
            network_font_size=0,
            node_size=4,
        )
        copy_uxsim_output(sim_outdir, OUTPUT_DIR,
                          f"network1_{t}.png",
                          f"network_snapshot_{label}.png")

    # ------------------------------------------------------------------
    # 3. Time-space trajectory diagram for corridor 0
    # ------------------------------------------------------------------
    corridor_id = 0
    corridor_links = links_by_corridor[corridor_id]
    link_names_joined = "-".join(l.name for l in corridor_links)

    print(f"Generating time-space trajectory diagram (corridor {corridor_id}) ...")
    W.analyzer.time_space_diagram_traj_links(corridor_links, figsize=(12, 5))
    copy_uxsim_output(sim_outdir, OUTPUT_DIR,
                      f"tsd_traj_links_{link_names_joined}.png",
                      "time_space_trajectory.png")

    # ------------------------------------------------------------------
    # 4. Time-space density diagram for corridor 0
    # ------------------------------------------------------------------
    print(f"Generating time-space density diagram (corridor {corridor_id}) ...")
    for link in corridor_links:
        W.analyzer.time_space_diagram_density([link], figsize=(12, 5))
    copy_uxsim_output(sim_outdir, OUTPUT_DIR,
                      f"tsd_k_{corridor_links[0].name}.png",
                      "time_space_density.png")

    # ------------------------------------------------------------------
    # 5. Cumulative arrival/departure curves for one link in corridor 0
    # ------------------------------------------------------------------
    print(f"Generating cumulative curves (corridor {corridor_id}) ...")
    # cumulative_curves produces one file per link
    W.analyzer.cumulative_curves([corridor_links[0]], figsize=(8, 5))
    # Note: UXsim uses "cumlative" (typo) in its filename
    copy_uxsim_output(sim_outdir, OUTPUT_DIR,
                      f"cumlative_curves_{corridor_links[0].name}.png",
                      "cumulative_curves.png")

    # ------------------------------------------------------------------
    # 6. Macroscopic Fundamental Diagram (MFD)
    # ------------------------------------------------------------------
    print("Generating macroscopic fundamental diagram ...")
    W.analyzer.macroscopic_fundamental_diagram(figsize=(6, 5))
    copy_uxsim_output(sim_outdir, OUTPUT_DIR,
                      "mfd.png", "mfd.png")

    # Clean up UXsim's temporary output directory
    if os.path.exists(sim_outdir):
        shutil.rmtree(sim_outdir)
        print(f"\nCleaned up {sim_outdir}/")

    print(f"All visualizations saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
