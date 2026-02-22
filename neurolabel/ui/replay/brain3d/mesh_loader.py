from __future__ import annotations

from pathlib import Path

import numpy as np

from .schemas import BrainMesh


class BrainMeshLoadError(RuntimeError):
    pass


def _generate_fallback_brain_mesh(*, nu: int = 88, nv: int = 58) -> BrainMesh:
    """Procedural cortical-looking fallback with folds and a midline fissure.

    This is not anatomical data; it is a visual proxy that reads as a brain (gyri/sulci)
    when fsaverage is unavailable.
    """
    vertices: list[list[float]] = []
    faces: list[list[int]] = []

    # Ellipsoid radii tuned for a brain-ish shape, y is anterior/posterior axis.
    rx, ry, rz = 1.0, 1.18, 0.92
    for j in range(nv + 1):
        v = j / nv
        phi = np.pi * v
        for i in range(nu):
            u = i / nu
            theta = 2 * np.pi * u
            sphi = np.sin(phi)
            cphi = np.cos(phi)
            cth = np.cos(theta)
            sth = np.sin(theta)

            # Layered harmonic ripples to evoke sulci/gyri on a coarse mesh.
            fold_env = (sphi ** 1.8)
            ripples = (
                0.055 * fold_env * np.sin(5.0 * theta + 1.1 * np.cos(3.0 * phi))
                + 0.030 * (sphi ** 2.3) * np.sin(11.0 * theta - 4.2 * phi + 0.6)
                + 0.018 * (sphi ** 2.6) * np.sin(17.0 * theta + 7.5 * phi)
            )
            r_scale = 1.0 + ripples

            x = rx * r_scale * sphi * cth
            y = ry * (1.0 + 0.7 * ripples) * sphi * sth
            z = rz * (1.0 + 0.45 * ripples) * cphi

            # Flatten underside slightly and add frontal bump for nicer silhouette.
            if z < -0.1:
                z *= 0.85
            y += 0.06 * np.exp(-((x / 0.65) ** 2 + ((z - 0.15) / 0.5) ** 2))

            # Longitudinal fissure between hemispheres (top/front weighted).
            top_weight = np.clip((z + 0.05) / 0.9, 0.0, 1.0)
            front_weight = np.clip((y + 0.1) / 1.3, 0.0, 1.0)
            fissure = 0.085 * np.exp(-((x / 0.11) ** 2)) * top_weight * front_weight
            z -= 0.08 * fissure
            y -= 0.03 * fissure

            # Subtle hemisphere fullness to avoid egg symmetry.
            x += 0.015 * np.sign(x) * (sphi ** 2.0) * top_weight
            vertices.append([float(x), float(y), float(z)])

    def idx(i: int, j: int) -> int:
        return j * nu + (i % nu)

    for j in range(nv):
        for i in range(nu):
            a = idx(i, j)
            b = idx(i + 1, j)
            c = idx(i, j + 1)
            d = idx(i + 1, j + 1)
            if j != 0:
                faces.append([a, c, b])
            if j != nv - 1:
                faces.append([b, c, d])

    return BrainMesh(
        source='procedural_fallback',
        vertices=np.asarray(vertices, dtype=np.float32),
        faces=np.asarray(faces, dtype=np.int32),
        units='arb',
    )


def _load_fsaverage_mesh() -> BrainMesh:
    try:
        import mne
    except Exception as exc:  # pragma: no cover - optional dependency
        raise BrainMeshLoadError('mne is not installed') from exc

    try:
        fs_dir = Path(mne.datasets.fetch_fsaverage(verbose=False))
        bem_dir = fs_dir / 'bem'
        src = None
        used_source_space = False

        # Prefer a sharper ico-6 pial source space if available; generate/cache it once if missing.
        ico6_src_path = bem_dir / 'fsaverage-ico-6-pial-src.fif'
        ico5_src_path = bem_dir / 'fsaverage-ico-5-src.fif'
        if ico6_src_path.exists():
            src = mne.read_source_spaces(ico6_src_path, verbose=False)
        else:
            try:
                src = mne.setup_source_space(
                    'fsaverage',
                    spacing='ico6',
                    surface='pial',
                    subjects_dir=str(fs_dir.parent),
                    add_dist=False,
                    verbose=False,
                )
                try:  # cache for future runs
                    mne.write_source_spaces(str(ico6_src_path), src, overwrite=True, verbose=False)
                except Exception:
                    pass
            except Exception:
                src = None

        # Fall back to shipped ico-5 source space if ico-6 creation is unavailable.
        if src is None and ico5_src_path.exists():
            src = mne.read_source_spaces(ico5_src_path, verbose=False)

        if src is not None:
            used_source_space = True
            if len(src) >= 2:
                lh = src[0]
                rh = src[1]
                lh_rr = np.asarray(lh['rr'][lh['vertno']], dtype=np.float32)
                rh_rr = np.asarray(rh['rr'][rh['vertno']], dtype=np.float32)
                lh_tris = np.asarray(lh['use_tris'], dtype=np.int32)
                rh_tris = np.asarray(rh['use_tris'], dtype=np.int32) + lh_rr.shape[0]
                vertices = np.vstack([lh_rr, rh_rr])
                faces = np.vstack([lh_tris, rh_tris])
            else:
                raise RuntimeError('fsaverage source space missing hemispheres')
        else:
            surf_dir = fs_dir / 'surf'
            lh_rr, lh_tris = mne.read_surface(str(surf_dir / 'lh.pial'), verbose=False)
            rh_rr, rh_tris = mne.read_surface(str(surf_dir / 'rh.pial'), verbose=False)
            lh_rr = np.asarray(lh_rr, dtype=np.float32)
            rh_rr = np.asarray(rh_rr, dtype=np.float32)
            lh_tris = np.asarray(lh_tris, dtype=np.int32)
            rh_tris = np.asarray(rh_tris, dtype=np.int32) + lh_rr.shape[0]
            vertices = np.vstack([lh_rr, rh_rr])
            faces = np.vstack([lh_tris, rh_tris])
    except Exception as exc:  # pragma: no cover - depends on env/network/cache
        raise BrainMeshLoadError(f'failed to load fsaverage mesh: {exc}') from exc

    # Normalize to manageable scale and orient to x=left/right, y=front/back, z=up heuristically.
    vertices = vertices - vertices.mean(axis=0, keepdims=True)
    max_abs = np.max(np.abs(vertices)) or 1.0
    vertices = (vertices / max_abs).astype(np.float32)

    # MNE/Freesurfer coordinates are not guaranteed to align with our viewer assumptions.
    # Swap y/z for the fNIRS hero-view orientation (equivalent to the matrix:
    # [1 0 0; 0 0 1; 0 1 0]).
    #
    # This transform has determinant -1 (a reflection), so it flips winding order.
    # Flip triangle winding too, otherwise front-face rendering produces a "sliced shell"
    # look from many camera angles.
    vertices = vertices[:, [0, 2, 1]]
    faces = faces[:, [0, 2, 1]]

    # If we fell back to full pial surfaces, cap payload with a conservative coherent
    # vertex-mask thinning (not random face skipping).
    if (not used_source_space) and faces.shape[0] > 120_000:
        keep = np.zeros(vertices.shape[0], dtype=bool)
        keep[::4] = True
        face_keep = keep[faces].all(axis=1)
        faces = faces[face_keep]
        used = np.unique(faces.reshape(-1))
        remap = -np.ones(vertices.shape[0], dtype=np.int32)
        remap[used] = np.arange(used.shape[0], dtype=np.int32)
        faces = remap[faces]
        vertices = vertices[used]

    source_name = 'mne_fsaverage'
    if faces.shape[0] >= 160_000:
        source_name = 'mne_fsaverage_ico6'
    return BrainMesh(source=source_name, vertices=vertices, faces=faces.astype(np.int32), units='norm')


def load_brain_mesh(*, prefer_fsaverage: bool = True) -> BrainMesh:
    if prefer_fsaverage:
        try:
            return _load_fsaverage_mesh()
        except BrainMeshLoadError:
            pass
    return _generate_fallback_brain_mesh()
