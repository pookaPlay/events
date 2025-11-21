"""
Convert an events NPZ (with keys like 'ev_loc', 'evs_norm', 'ev')
into the JSON format expected by `test_syn.py` / `test_syn_npz.py`.

Output JSON structure:
{
  "totalFrames": N,
  "frames": [ {"events": [{"x":..., "y":..., "t":..., "p":...}, ...] }, ... ]
}

Usage:
    python npz_to_json.py input.npz output.json [--num-frames N]

The script will try a few sensible heuristics to determine polarity (`p`) values
and will map {0,1} -> {-1,1} if needed. Use `--num-frames` to aggregate NPZ
time indices into fewer JSON frames: events with integer time indices
`0:(N-1)` -> frame 0, `N:(2*N-1)` -> frame 1, etc.
"""
import argparse
import json
from pathlib import Path
import numpy as np


def determine_polarities(npz):
    # Try multiple sources for polarity
    # 1) If 'ev' exists and is numeric with >=4 columns, use column 3
    if 'ev' in npz:
        ev = npz['ev']
        if getattr(ev, 'ndim', 0) == 2 and ev.shape[1] >= 4:
            return ev[:, 3].astype(np.float32)
        # handle structured array with fields
        if ev.dtype.names and 'p' in ev.dtype.names:
            return ev['p'].astype(np.float32)

    # 2) If 'evs_norm' exists and has >=4 columns, use column 3
    if 'evs_norm' in npz:
        norm = npz['evs_norm']
        if getattr(norm, 'ndim', 0) == 2 and norm.shape[1] >= 4:
            return norm[:, 3].astype(np.float32)

    # 3) Fallback: no explicit polarity — assume positive polarity (1)
    n = len(npz[next(iter(npz.files))])
    return np.ones((n,), dtype=np.float32)


def map_p_values(p_array):
    # Convert float array to integer polarities -1 or 1
    unique = np.unique(p_array)
    # If values are 0/1 -> map 0->-1, 1->1
    if set(np.isin(unique, [0, 1])) == {True}:
        p = np.where(p_array == 0, -1, 1)
        return p.astype(int)
    # If values already -1/1 or close, round to nearest int
    p_rounded = np.rint(p_array).astype(int)
    # Ensure values are -1 or 1, otherwise map non-positive to -1
    p_final = np.where(p_rounded <= 0, -1, 1)
    return p_final


def build_frames(x, y, t, p, num_frames=1):
    # group by integer frame index t, then aggregate by `num_frames` into output frames
    if num_frames < 1:
        raise ValueError("num_frames must be >= 1")

    t_int = t.astype(int)
    # map each event time to an output frame index by integer division
    frame_idx = (t_int // int(num_frames)).astype(int)
    max_frame = int(frame_idx.max()) if frame_idx.size > 0 else -1
    total_frames = max_frame + 1
    frames = []
    for fi in range(total_frames):
        mask = (frame_idx == fi)
        events = []
        for xi, yi, ti, pi in zip(x[mask], y[mask], t[mask], p[mask]):
            events.append({
                'x': int(xi),
                'y': int(yi),
                't': float(ti),
                'p': int(pi)
            })
        frames.append({'events': events})
    return total_frames, frames


def main():
    parser = argparse.ArgumentParser(description='Convert events NPZ to JSON frames format')
    parser.add_argument('input', type=str, help='Input .npz file')
    parser.add_argument('output', type=str, help='Output .json file')
    parser.add_argument('--num-frames', type=int, default=10,
                        help='Number of NPZ time indices to group per JSON frame')
    args = parser.parse_args()

    inp = Path(args.input)
    outp = Path(args.output)
    if not inp.exists():
        raise SystemExit(f"Input file not found: {inp}")

    data = np.load(inp, allow_pickle=True)

    # Basic info
    print("Loaded NPZ. Keys:", data.files)

    # Required: ev_loc -> expected shape (N,3) : x,y,t
    if 'ev_loc' not in data:
        raise SystemExit("NPZ must contain 'ev_loc' with x,y,t per-event locations")

    ev_loc = data['ev_loc']
    if ev_loc.ndim != 2 or ev_loc.shape[1] < 3:
        raise SystemExit("'ev_loc' must be shape (N,3) with columns x,y,t")

    x = ev_loc[:, 0].astype(int)
    y = ev_loc[:, 1].astype(int)
    t = ev_loc[:, 2].astype(float)

    raw_p = determine_polarities(data)
    p = map_p_values(raw_p)

    total_frames, frames = build_frames(x, y, t, p, num_frames=args.num_frames)

    out = {
        'totalFrames': total_frames,
        'frames': frames
    }

    with open(outp, 'w') as f:
        json.dump(out, f)

    print(f"Wrote JSON with {total_frames} frames to {outp}")


if __name__ == '__main__':
    main()
