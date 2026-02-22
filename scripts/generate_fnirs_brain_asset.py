#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import trimesh

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from neurolabel.ui.replay.brain3d.mesh_loader import load_brain_mesh
from neurolabel.ui.replay.brain3d.pfc_mapping import build_pfc_proxy_mapping


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate high-fidelity fNIRS brain display asset + mapping JSON.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("web/frontend/public/models"),
        help="Output directory for brain.glb and brain_fnirs_mapping.json",
    )
    parser.add_argument(
        "--asset-id",
        default="brain_glb_v1",
        help="Identifier embedded in the mapping JSON",
    )
    args = parser.parse_args()

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    mesh = load_brain_mesh(prefer_fsaverage=True)
    mapping = build_pfc_proxy_mapping(mesh)

    glb_path = out_dir / "brain.glb"
    mapping_path = out_dir / "brain_fnirs_mapping.json"

    tri = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, process=False)
    glb_bytes = tri.export(file_type="glb")
    glb_path.write_bytes(glb_bytes)

    mapping_json = {
        "schema_version": 1,
        "asset_id": args.asset_id,
        "mesh_source": mesh.source,
        "vertex_count": int(mesh.vertices.shape[0]),
        "face_count": int(mesh.faces.shape[0]),
        "roi_mask": mapping.roi_mask.astype(int).tolist(),
        "left_weights": mapping.left_weights.astype(float).tolist(),
        "right_weights": mapping.right_weights.astype(float).tolist(),
        "anchors": mapping.anchors,
        "orientation": {
            "scale": 1.24,
            "rotation_euler_xyz": [-0.12, 0.18, 1.48],
            "translation": [0.0, -0.02, -0.12],
        },
        "notes": {
            "description": "Proxy PFC heatmap weights for the display brain mesh. Sparse fNIRS sensors, not volumetric imaging.",
            "x_axis": "left/right",
            "y_axis": "front/back",
            "z_axis": "up/down",
        },
    }
    mapping_path.write_text(json.dumps(mapping_json), encoding="utf-8")

    print(f"Wrote {glb_path} ({glb_path.stat().st_size} bytes)")
    print(f"Wrote {mapping_path} ({mapping_path.stat().st_size} bytes)")
    print(f"Mesh source={mesh.source} verts={mesh.vertices.shape[0]} faces={mesh.faces.shape[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
