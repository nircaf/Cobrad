"""EEG channel values -> rendered 3D brain heat-map, via Blender.

Public entry point:

    render_eeg_brain({"Fp1": 12.4, "Fp2": 9.8, "Cz": 3.1, ...}, out_path="eeg.png")

Inside a running Blender (this is what blender-mcp drives):

    exec(open(r"C:\\Nir\\dreams\\Arduino\\stimx\\tools\\eeg_brain_blender.py").read())
    render_eeg_brain(demo_channels(), out_path=r"C:\\temp\\eeg_brain.png")

Headless:

    blender -b -P tools/eeg_brain_blender.py -- channels.json out.png

The JSON is either a bare {label: value} map, or a config object that also sets
render options (view defaults to "top"):

    {
      "channels": {"Fp1": 41.0, "Fz": 36.5, "Cz": 25.0, "Oz": 9.0},
      "view": "right_lateral",        // a VIEWS name, or a direction [x, y, z]
      "cmap": "red_white_blue",       // a CMAPS name, or [[pos, [r,g,b]], ...]
      "vmin": -40, "vmax": 40,
      "fill": 0.80
    }

Video, with values changing over time and the camera moving between angles:

    render_eeg_brain_video(
        {"Fp1": [12.0, 40.0, 18.0], "Oz": [8.0, 8.0, 30.0]},  # ramps over the clip
        out_path=r"C:\\temp\\eeg_brain.mp4",
        views=[(0.0, "top"), (0.5, "right_lateral"), (1.0, "anterior")],
        duration=6, fps=24,
    )

Self-check of the hardware-free math (plain python + numpy, no Blender):

    python tools/eeg_brain_blender.py --self-check
"""

import json
import math
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

try:
    import bpy
    from mathutils import Matrix, Vector, noise
except ImportError:  # importable outside Blender so the math stays testable
    bpy = None


# --- montage -----------------------------------------------------------------
# 10-20 / 10-10 electrodes as EEGLAB-style topoplot polar coords:
# (theta_deg clockwise from nose, r where 0 = Cz and 0.5 = the Fpz-T7-Oz-T8 equator).
# The 10-10 intermediates (AF*, FC*, CP*, PO*) are interpolated, not tabulated —
# good enough for a visualization, not for source localization.
MONTAGE_POLAR = {
    "Fp1": (-18, 0.511), "Fpz": (0, 0.511), "Fp2": (18, 0.511),
    "AF7": (-36, 0.480), "AF3": (-24, 0.420), "AFz": (0, 0.380),
    "AF4": (24, 0.420), "AF8": (36, 0.480),
    "F7": (-54, 0.511), "F3": (-39, 0.333), "Fz": (0, 0.256),
    "F4": (39, 0.333), "F8": (54, 0.511),
    "FC5": (-69, 0.430), "FC1": (-45, 0.200), "FC2": (45, 0.200), "FC6": (69, 0.430),
    "T7": (-90, 0.511), "T3": (-90, 0.511), "C3": (-90, 0.256), "Cz": (0, 0.0),
    "C4": (90, 0.256), "T8": (90, 0.511), "T4": (90, 0.511),
    "CP5": (-111, 0.430), "CP1": (-135, 0.200), "CP2": (135, 0.200), "CP6": (111, 0.430),
    "P7": (-126, 0.511), "T5": (-126, 0.511), "P3": (-141, 0.333), "Pz": (180, 0.256),
    "P4": (141, 0.333), "P8": (126, 0.511), "T6": (126, 0.511),
    "PO7": (-144, 0.480), "PO3": (-153, 0.400), "POz": (180, 0.380),
    "PO4": (153, 0.400), "PO8": (144, 0.480),
    "O1": (-162, 0.511), "Oz": (180, 0.511), "O2": (162, 0.511),
    "A1": (-108, 0.510), "A2": (108, 0.510),
}

# Named ramps as [(position in 0..1, (r, g, b)), ...]. Pass a name or your own list
# of stops as `cmap`; positions need not be sorted.
CMAPS = {
    # perceptually-rising cool -> warm (deep blue, cyan, teal, gold, red)
    "spectral": [
        (0.00, (0.030, 0.070, 0.290)),
        (0.25, (0.070, 0.480, 0.850)),
        (0.50, (0.130, 0.830, 0.680)),
        (0.72, (0.960, 0.810, 0.230)),
        (1.00, (0.950, 0.190, 0.160)),
    ],
    # diverging: low = red, mid = white, high = blue
    "red_white_blue": [
        (0.00, (0.870, 0.130, 0.110)),
        (0.50, (1.000, 1.000, 1.000)),
        (1.00, (0.110, 0.330, 0.880)),
    ],
    # single-hue, dark -> bright, for "more is brighter" without a hue story
    "ember": [
        (0.00, (0.050, 0.020, 0.080)),
        (0.45, (0.640, 0.120, 0.260)),
        (0.78, (0.960, 0.520, 0.150)),
        (1.00, (1.000, 0.940, 0.750)),
    ],
}
DEFAULT_CMAP = "spectral"


def resolve_cmap(cmap=None):
    """None | name | [(pos, (r,g,b)), ...] -> validated, position-sorted stop list."""
    if cmap is None:
        cmap = DEFAULT_CMAP
    if isinstance(cmap, str):
        try:
            return CMAPS[cmap]
        except KeyError:
            raise ValueError(f"unknown cmap {cmap!r}; known: {', '.join(sorted(CMAPS))}")
    stops = []
    for stop in cmap:
        pos, rgb = stop
        if len(rgb) != 3:
            raise ValueError(f"cmap stop {stop!r} needs an (r, g, b) colour")
        stops.append((float(pos), tuple(float(c) for c in rgb)))
    if len(stops) < 2:
        raise ValueError("cmap needs at least 2 stops")
    return sorted(stops)  # np.interp requires increasing positions

BRAIN_SCALE = (0.80, 1.05, 0.78)


def electrode_dir(name):
    """Unit vector for a montage label. +x = right, +y = anterior, +z = superior."""
    try:
        theta, r = MONTAGE_POLAR[name]
    except KeyError:
        raise ValueError(
            f"unknown EEG channel {name!r}; known: {', '.join(sorted(MONTAGE_POLAR))}"
        )
    el = math.radians(90.0 - 180.0 * r)
    t = math.radians(theta)
    ce = math.cos(el)
    return np.array([ce * math.sin(t), ce * math.cos(t), math.sin(el)])


def colormap(t, cmap=None):
    """[0,1] -> RGB, shape (..., 3). cmap: name, stop list, or None for the default."""
    stops = resolve_cmap(cmap)
    t = np.clip(np.asarray(t, dtype=float), 0.0, 1.0)
    pos = np.array([s[0] for s in stops])
    cols = np.array([s[1] for s in stops])
    return np.stack([np.interp(t, pos, cols[:, i]) for i in range(3)], axis=-1)


def scalp_interpolate(dirs, elec_dirs, values, sigma=0.42):
    """Gaussian-RBF spread of per-channel values over unit directions `dirs`.

    sigma is in radians of great-circle distance; smaller = tighter hot spots.
    """
    cos = np.clip(np.asarray(dirs) @ np.asarray(elec_dirs).T, -1.0, 1.0)
    w = np.exp(-((np.arccos(cos) / sigma) ** 2))
    w = np.maximum(w, 1e-9)
    return (w @ np.asarray(values, dtype=float)) / w.sum(axis=1)


def normalize(values, vmin=None, vmax=None):
    v = np.asarray(values, dtype=float)
    lo = float(np.min(v)) if vmin is None else float(vmin)
    hi = float(np.max(v)) if vmax is None else float(vmax)
    if hi - lo < 1e-12:  # flat input: render mid-scale instead of dividing by ~0
        return np.full_like(v, 0.5), lo, hi
    return (v - lo) / (hi - lo), lo, hi


def demo_channels():
    """Plausible frontal-dominant slow-wave map, for eyeballing the render."""
    return {
        "Fp1": 41.0, "Fp2": 38.5, "F7": 26.0, "F3": 34.0, "Fz": 36.5, "F4": 31.0,
        "F8": 24.0, "T7": 14.0, "C3": 22.0, "Cz": 25.0, "C4": 20.5, "T8": 13.0,
        "P7": 11.0, "P3": 16.0, "Pz": 18.0, "P4": 15.0, "P8": 10.0,
        "O1": 8.5, "Oz": 9.0, "O2": 8.0,
    }


def demo_channel_series():
    """A frontal hot spot that fades out while an occipital one rises, for the video."""
    base = demo_channels()
    return {name: [v, v * 0.35, v * 1.4] if name.startswith(("F", "Fp"))
            else [v, v * 1.6, v * 0.5] for name, v in base.items()}


def _series_at(channel_series, t):
    """channel_series values -> {label: value} at fraction t in [0, 1].

    Each channel's own samples are assumed evenly spaced across the full clip
    and linearly interpolated; a single-sample channel stays constant.
    """
    out = {}
    for name, vals in channel_series.items():
        vals = np.atleast_1d(np.asarray(vals, dtype=float))
        out[name] = float(vals[0]) if vals.size == 1 else \
            float(np.interp(t, np.linspace(0.0, 1.0, vals.size), vals))
    return out


# --- Blender scene -----------------------------------------------------------
COLLECTION = "EEG_Brain"
_STARTUP_JUNK = {"Cube", "Light", "Camera"}


def _collection(clear=True):
    col = bpy.data.collections.get(COLLECTION)
    if col is None:
        col = bpy.data.collections.new(COLLECTION)
        bpy.context.scene.collection.children.link(col)
    elif clear:
        for ob in list(col.objects):
            bpy.data.objects.remove(ob, do_unlink=True)
    return col


def _link_only(ob, col):
    for c in list(ob.users_collection):
        c.objects.unlink(ob)
    col.objects.link(ob)


def _socket(node, names, value):
    for n in names:
        if n in node.inputs:
            node.inputs[n].default_value = value
            return


def _deform(x, y, z, lift=1.0):
    """Unit sphere direction -> point on the cortex surface. lift>1 floats above it."""
    # |sin(fractal)| bands follow the noise iso-contours -> winding gyri/sulci.
    v = Vector((x, y, z))
    g = noise.fractal(v * 2.4, 1.0, 2.0, 4)
    warp = noise.noise(v * 6.1)  # breaks the concentric iso-contour bullseyes
    k = 1.0 - 0.058 * (1.0 - abs(math.sin(g * 5.0 + warp * 2.4)))
    k -= 0.085 * math.exp(-((x / 0.055) ** 2)) * math.sqrt(max(0.0, z))  # midline fissure
    k -= 0.055 * math.exp(-(((z + 0.16 - 0.30 * y) / 0.10) ** 2)) \
        * min(1.0, 2.0 * x * x)                                          # Sylvian fissure
    k += 0.050 * math.exp(-(((z + 0.46) / 0.30) ** 2)) \
        * min(1.0, 2.0 * x * x)                                          # temporal lobe
    taper = 1.0 - 0.12 * max(0.0, y)  # frontal lobe is narrower than parietal
    px = x * BRAIN_SCALE[0] * k * taper
    py = y * BRAIN_SCALE[1] * k
    pz = z * BRAIN_SCALE[2] * k
    if pz < 0.0:
        pz *= 0.62  # brains are flat-ish inferiorly
    return px * lift, py * lift, pz * lift


def _brain_mesh(col, subdiv=6):
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=subdiv, radius=1.0)
    ob = bpy.context.active_object
    ob.name = "EEG_Brain_Surface"
    _link_only(ob, col)

    mesh = ob.data
    n = len(mesh.vertices)
    co = np.empty(n * 3, dtype=np.float64)
    mesh.vertices.foreach_get("co", co)
    dirs = co.reshape(-1, 3).copy()
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)

    out = np.empty_like(dirs)
    for i in range(n):
        out[i] = _deform(float(dirs[i][0]), float(dirs[i][1]), float(dirs[i][2]))

    mesh.vertices.foreach_set("co", out.ravel())
    mesh.polygons.foreach_set("use_smooth", [True] * len(mesh.polygons))
    mesh.update()
    return ob, dirs


def _brain_material(name="EEG_Cortex", attr="EEG"):
    mat = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    out.location = (400, 0)
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (100, 0)
    a = nt.nodes.new("ShaderNodeAttribute")
    a.attribute_name = attr
    a.location = (-250, 0)
    nt.links.new(a.outputs["Color"], bsdf.inputs["Base Color"])
    if "Emission Color" in bsdf.inputs:
        nt.links.new(a.outputs["Color"], bsdf.inputs["Emission Color"])
    elif "Emission" in bsdf.inputs:
        nt.links.new(a.outputs["Color"], bsdf.inputs["Emission"])
    _socket(bsdf, ["Emission Strength"], 0.08)  # a lift off pure black, not a light source
    _socket(bsdf, ["Roughness"], 0.52)  # lower = a specular haze over the value ramp
    _socket(bsdf, ["Specular IOR Level", "Specular"], 0.30)
    _socket(bsdf, ["Subsurface Weight"], 0.15)
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return mat


def _electrode_material(name="EEG_Electrode"):
    """One emission material for every probe; per-object tint via Object Info -> Color."""
    mat = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs["Strength"].default_value = 1.6  # higher clips the tint to white
    info = nt.nodes.new("ShaderNodeObjectInfo")
    nt.links.new(info.outputs["Color"], em.inputs["Color"])
    nt.links.new(em.outputs["Emission"], out.inputs["Surface"])
    return mat


def _lights(col):
    for name, loc, energy, color, size in [
        # keep the sum well under clipping or the value colours wash out to white
        ("EEG_Key", (2.6, 2.2, 2.8), 110.0, (1.0, 0.96, 0.92), 2.4),
        ("EEG_Fill", (-2.9, 0.6, 0.5), 45.0, (0.62, 0.78, 1.0), 3.5),
        ("EEG_Rim", (-0.6, -3.2, 1.6), 85.0, (1.0, 0.72, 0.55), 2.0),
    ]:
        d = bpy.data.lights.new(name, type="AREA")
        d.energy = energy
        d.color = color
        d.size = size
        ob = bpy.data.objects.new(name, d)
        ob.location = loc
        ob.rotation_euler = (-Vector(loc)).to_track_quat("-Z", "Y").to_euler()
        col.objects.link(ob)


# Camera *directions* (+x right, +y anterior, +z superior). Distance is solved per
# render by frame_distance(), so any direction frames the brain identically.
VIEWS = {
    "top": (0.0, 0.0, 1.0),
    "anterior_right": (0.62, 0.60, 0.40),
    "right_lateral": (1.0, 0.07, 0.25),
    "left_lateral": (-1.0, 0.07, 0.25),
    "posterior_left": (-0.60, -0.62, 0.40),
    "anterior": (0.0, 1.0, 0.22),
    # Fp1/Fp2/Fpz sit at r=0.511, the equator, so they are edge-on from "top" and
    # only half-visible from "anterior"; this oblique faces the frontal pole.
    "anterior_superior": (0.0, 1.0, 0.75),
}

SENSOR_MM = 36.0  # Blender's default camera sensor width


def _slerp_dir(a, b, t):
    """Great-circle interpolation between two directions at t in [0, 1]."""
    a = np.asarray(a, dtype=float); a = a / np.linalg.norm(a)
    b = np.asarray(b, dtype=float); b = b / np.linalg.norm(b)
    theta = math.acos(float(np.clip(a @ b, -1.0, 1.0)))
    if theta < 1e-6:
        return a
    s = math.sin(theta)
    return (math.sin((1.0 - t) * theta) / s) * a + (math.sin(t * theta) / s) * b


def _resolve_view_keyframes(views):
    """view(s) -> sorted [(t, unit direction), ...].

    Accepts a single VIEWS name, a single (x, y, z) direction, or a list of
    (t, view) pairs (t in [0, 1]) to slerp the camera between over the clip.
    """
    if isinstance(views, str) or (
        len(views) == 3 and all(isinstance(c, (int, float)) for c in views)
    ):
        views = [(0.0, views)]
    kfs = []
    for t, v in views:
        if isinstance(v, str):
            try:
                v = VIEWS[v]
            except KeyError:
                raise ValueError(f"unknown view {v!r}; known: {', '.join(VIEWS)}")
        v = np.asarray(v, dtype=float)
        if np.linalg.norm(v) < 1e-9:
            raise ValueError("view direction cannot be a zero vector")
        kfs.append((float(t), v / np.linalg.norm(v)))
    if not kfs:
        raise ValueError("views needs at least one keyframe")
    return sorted(kfs, key=lambda kv: kv[0])


def _direction_at(view_kfs, t):
    """Interpolated camera direction at fraction t, holding at the endpoints."""
    direction = view_kfs[0][1]
    for (t0, d0), (t1, d1) in zip(view_kfs, view_kfs[1:]):
        if t <= t1:
            local = 0.0 if t1 <= t0 else float(np.clip((t - t0) / (t1 - t0), 0.0, 1.0))
            return _slerp_dir(d0, d1, local)
        direction = d1
    return direction


def camera_basis(direction):
    """Right-handed (right, up, fwd) for a camera looking back down `direction`.

    fwd points from the subject toward the camera, i.e. the camera's local +Z.
    """
    fwd = np.asarray(direction, dtype=float)
    norm = np.linalg.norm(fwd)
    if norm < 1e-9:
        raise ValueError("view direction cannot be a zero vector")
    fwd = fwd / norm
    # world +Z is the natural up, except looking straight down it is degenerate;
    # there, +Y (anterior) up gives the conventional nose-up topographic layout.
    up_hint = np.array([0.0, 1.0, 0.0]) if abs(fwd[2]) > 0.9 else np.array([0.0, 0.0, 1.0])
    right = np.cross(up_hint, fwd)
    right /= np.linalg.norm(right)
    return right, np.cross(fwd, right), fwd


def frame_distance(verts, center, basis, tan_h, tan_v, fill=0.80):
    """Smallest camera distance from `center` that keeps every vertex inside `fill`.

    Solved per-vertex against its own depth, so foreshortened geometry near the
    camera cannot poke out of frame the way a bounding-sphere estimate allows.
    """
    right, up, fwd = basis
    rel = np.asarray(verts, dtype=float) - np.asarray(center, dtype=float)
    need = (rel @ fwd) + np.maximum(np.abs(rel @ right) / (fill * tan_h),
                                    np.abs(rel @ up) / (fill * tan_v))
    return float(need.max())


def _place_camera(cam, verts, center, direction, lens, fill, resolution):
    """Point an existing camera object at `direction`, framing `verts` around `center`."""
    basis = camera_basis(direction)
    right, up, fwd = basis
    res_x, res_y = resolution
    tan_h = (SENSOR_MM / 2.0) / lens
    tan_v = tan_h * res_y / res_x
    cam.location = center + fwd * frame_distance(verts, center, basis, tan_h, tan_v, fill)
    # Build the orientation from the same basis rather than to_track_quat(), which
    # has no defined roll looking straight down.
    cam.rotation_euler = Matrix(((right[0], up[0], fwd[0]),
                                 (right[1], up[1], fwd[1]),
                                 (right[2], up[2], fwd[2]))).to_euler()


def _mesh_vertex_array(ob):
    n = len(ob.data.vertices)
    co = np.empty(n * 3, dtype=np.float64)
    ob.data.vertices.foreach_get("co", co)
    return co.reshape(-1, 3)  # the surface is built untransformed at the origin


def _frame_camera(col, ob, direction, lens=62.0, fill=0.80, resolution=(1600, 1000)):
    v = _mesh_vertex_array(ob)
    center = 0.5 * (v.min(axis=0) + v.max(axis=0))

    d = bpy.data.cameras.new("EEG_Camera")
    d.lens = lens
    d.sensor_width = SENSOR_MM
    d.sensor_fit = "HORIZONTAL"  # frame_distance assumes the fit axis is x
    cam = bpy.data.objects.new("EEG_Camera", d)
    col.objects.link(cam)
    bpy.context.scene.camera = cam
    _place_camera(cam, v, center, direction, lens, fill, resolution)
    return cam


def render_eeg_brain(channels, out_path=None, vmin=None, vmax=None, sigma=0.42,
                     subdiv=8, samples=192, resolution=(1600, 1000),
                     clear_scene=True, show_electrodes=False, view="top",
                     cmap=None, denoise=False, fill=0.80):
    """Build and render a brain surface coloured by per-channel EEG values.

    channels: {"Fp1": value, ...} using MONTAGE_POLAR labels.
    view: a VIEWS key, or an (x, y, z) direction to look from (length ignored).
    fill: fraction of the frame the brain spans. The camera distance is solved
        from the mesh for whatever direction is given, so it frames the same
        from every angle.
    subdiv/samples/denoise: subdiv 8 + denoise off keeps the sulci crisp (~25 s);
        subdiv 6, samples 32, denoise=True is the fast preview (~3 s).
    cmap: a CMAPS key, or your own [(pos, (r, g, b)), ...] stops. Values are
        normalised to 0..1 via vmin/vmax first, so pass those to pin what a stop
        position means in your units (vmin=-1, vmax=1 puts 0 at the 0.5 stop).
    Returns the output image path (None if out_path was omitted -> scene only).
    """
    if bpy is None:
        raise RuntimeError("render_eeg_brain() must run inside Blender")
    if not channels:
        raise ValueError("channels is empty")

    names = list(channels)
    elec_dirs = np.array([electrode_dir(n) for n in names])
    t, lo, hi = normalize([float(channels[n]) for n in names], vmin, vmax)
    stops = resolve_cmap(cmap)  # validate before building anything

    scene = bpy.context.scene
    if clear_scene:
        for ob in list(bpy.data.objects):
            if ob.name in _STARTUP_JUNK:
                bpy.data.objects.remove(ob, do_unlink=True)
    col = _collection(clear=True)

    brain, dirs = _brain_mesh(col, subdiv=subdiv)
    vt = scalp_interpolate(dirs, elec_dirs, t, sigma=sigma)
    rgba = np.ones((len(vt), 4))
    rgba[:, :3] = colormap(vt, stops)
    attr = brain.data.color_attributes.new(name="EEG", type="FLOAT_COLOR", domain="POINT")
    attr.data.foreach_set("color", rgba.ravel())
    brain.data.attributes.active_color = attr
    brain.data.materials.append(_brain_material())

    if show_electrodes:
        emat = _electrode_material()
        colors = colormap(t, stops)
        for name, d, ti, c in zip(names, elec_dirs, t, colors):
            bpy.ops.mesh.primitive_uv_sphere_add(segments=20, ring_count=12,
                                                 radius=0.024 + 0.028 * float(ti))
            ob = bpy.context.active_object
            ob.name = f"EEG_{name}"
            # same deform as the mesh, lifted clear of the sulci so it never sinks in
            ob.location = _deform(float(d[0]), float(d[1]), float(d[2]), lift=1.025)
            ob.color = (float(c[0]), float(c[1]), float(c[2]), 1.0)
            ob.data.materials.append(emat)
            for poly in ob.data.polygons:
                poly.use_smooth = True
            _link_only(ob, col)

    _lights(col)
    if isinstance(view, str):
        try:
            view = VIEWS[view]
        except KeyError:
            raise ValueError(f"unknown view {view!r}; known: {', '.join(VIEWS)}")
    _frame_camera(col, brain, view, fill=fill, resolution=resolution)

    world = scene.world or bpy.data.worlds.new("World")
    scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs[0].default_value = (0.010, 0.013, 0.022, 1.0)
        bg.inputs[1].default_value = 1.0

    # AgX/Filmic desaturate the highlights, which is exactly what a value ramp
    # must not do; Standard keeps the mapped colour readable.
    try:
        scene.view_settings.view_transform = "Standard"
    except TypeError:
        pass

    scene.render.engine = "CYCLES"
    scene.cycles.samples = samples
    # OIDN smooths away the sulcus edges — cheap previews only, off for finals.
    scene.cycles.use_denoising = denoise
    scene.render.resolution_x, scene.render.resolution_y = resolution
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"

    print(f"[eeg_brain] {len(names)} channels, range {lo:.3g}..{hi:.3g}")
    if out_path:
        scene.render.filepath = out_path
        bpy.ops.render.render(write_still=True)
    return out_path


def render_eeg_brain_video(channel_series, out_path, views="top", duration=None,
                           fps=24, vmin=None, vmax=None, cmap=None, sigma=0.42,
                           subdiv=7, samples=48, resolution=(1600, 1000),
                           fill=0.80, denoise=True, clear_scene=True):
    """Render an .mp4 of the cortex heat-map animating over time and camera angle.

    channel_series: {"Fp1": [v0, v1, ...], ...}. Each channel's samples are
        assumed evenly spaced across the whole clip and linearly interpolated
        per frame; a channel given a single value stays constant throughout.
    views: a VIEWS name / (x, y, z) direction held fixed for the whole clip, or
        a list of (t, view) keyframes with t in [0, 1] — the camera slerps
        between them and holds at the first/last direction outside that range.
    duration: seconds of video. Defaults to (max samples across channels - 1),
        i.e. one data keyframe per second, minimum 1s.
    subdiv/samples/denoise default lower/on relative to the still-image function
        because cost multiplies by frame count; raise subdiv/samples and turn
        denoise off for a crisper final pass once framing and timing look right.
    Renders a PNG sequence and muxes it with the standalone `ffmpeg` binary (most
    Blender builds, this one included, ship without the FFMPEG codec compiled
    in). Returns out_path on success; if ffmpeg isn't on PATH or muxing fails,
    prints why and returns the temp directory of PNG frames instead.
    """
    if bpy is None:
        raise RuntimeError("render_eeg_brain_video() must run inside Blender")
    if not channel_series:
        raise ValueError("channel_series is empty")

    names = list(channel_series)
    elec_dirs = np.array([electrode_dir(n) for n in names])
    stops = resolve_cmap(cmap)  # validate before building anything
    view_kfs = _resolve_view_keyframes(views)

    n_samples = max(np.atleast_1d(channel_series[n]).size for n in names)
    if duration is None:
        duration = max(1.0, float(n_samples - 1))
    n_frames = max(2, int(round(duration * fps)) + 1)

    all_vals = np.concatenate([np.atleast_1d(np.asarray(channel_series[n], dtype=float))
                               for n in names])
    _, lo, hi = normalize(all_vals, vmin, vmax)

    scene = bpy.context.scene
    if clear_scene:
        for ob in list(bpy.data.objects):
            if ob.name in _STARTUP_JUNK:
                bpy.data.objects.remove(ob, do_unlink=True)
    col = _collection(clear=True)

    brain, dirs = _brain_mesh(col, subdiv=subdiv)
    brain.data.materials.append(_brain_material())
    attr = brain.data.color_attributes.new(name="EEG", type="FLOAT_COLOR", domain="POINT")
    brain.data.attributes.active_color = attr

    verts = _mesh_vertex_array(brain)
    center = 0.5 * (verts.min(axis=0) + verts.max(axis=0))

    _lights(col)
    cam = _frame_camera(col, brain, view_kfs[0][1], fill=fill, resolution=resolution)

    world = scene.world or bpy.data.worlds.new("World")
    scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs[0].default_value = (0.010, 0.013, 0.022, 1.0)
        bg.inputs[1].default_value = 1.0
    try:
        scene.view_settings.view_transform = "Standard"
    except TypeError:
        pass

    scene.render.engine = "CYCLES"
    scene.cycles.samples = samples
    scene.cycles.use_denoising = denoise
    scene.render.resolution_x, scene.render.resolution_y = resolution
    scene.render.resolution_percentage = 100
    scene.render.fps = fps
    scene.frame_start = 1
    scene.frame_end = n_frames
    # ponytail: many Blender builds (this one included) ship without the FFMPEG
    # muxer compiled in, so render a PNG sequence and mux with the standalone
    # ffmpeg binary instead — works on every build. Upgrade to
    # scene.render.image_settings.file_format = "FFMPEG" directly if your build
    # has it and you want to skip the PNG round-trip.
    frame_dir = tempfile.mkdtemp(prefix="eeg_brain_frames_")
    scene.render.image_settings.file_format = "PNG"
    scene.render.filepath = os.path.join(frame_dir, "frame_")

    rgba = np.ones((len(dirs), 4))

    def _update(frame_no):
        span = max(1, scene.frame_end - scene.frame_start)
        t = float(np.clip((frame_no - scene.frame_start) / span, 0.0, 1.0))
        vals = _series_at(channel_series, t)
        vt = scalp_interpolate(dirs, elec_dirs, [vals[n] for n in names], sigma=sigma)
        tnorm = np.zeros_like(vt) + 0.5 if hi <= lo else np.clip((vt - lo) / (hi - lo), 0.0, 1.0)
        rgba[:, :3] = colormap(tnorm, stops)
        attr.data.foreach_set("color", rgba.ravel())
        _place_camera(cam, verts, center, _direction_at(view_kfs, t), cam.data.lens,
                     fill, resolution)

    def _handler(scene_, *_):
        _update(scene_.frame_current)

    _update(scene.frame_start)  # frame_change_pre is not guaranteed to fire for frame_start
    bpy.app.handlers.frame_change_pre.append(_handler)
    try:
        bpy.ops.render.render(animation=True)
    finally:
        bpy.app.handlers.frame_change_pre.remove(_handler)

    print(f"[eeg_brain] video: {n_frames} frames @ {fps}fps over {duration:.1f}s, "
          f"{len(names)} channels, range {lo:.3g}..{hi:.3g}")

    if shutil.which("ffmpeg") is None:
        print(f"[eeg_brain] ffmpeg not found on PATH; PNG frames left in {frame_dir}")
        return frame_dir
    result = subprocess.run(
        ["ffmpeg", "-y", "-framerate", str(fps), "-start_number", str(scene.frame_start),
         "-i", os.path.join(frame_dir, "frame_%04d.png"), "-pix_fmt", "yuv420p", out_path],
        capture_output=True,
    )
    if result.returncode != 0:
        print(f"[eeg_brain] ffmpeg mux failed; PNG frames left in {frame_dir}\n"
              f"{result.stderr.decode(errors='replace')[-500:]}")
        return frame_dir
    shutil.rmtree(frame_dir, ignore_errors=True)
    return out_path


_JSON_KEYS = {"channels", "out_path", "view", "cmap", "vmin", "vmax", "sigma",
              "fill", "subdiv", "samples", "denoise", "resolution",
              "show_electrodes", "clear_scene"}


def load_config(cfg):
    """Accept either a bare {label: value} map or a config object; validate keys."""
    if "channels" not in cfg:
        return {"channels": cfg}
    unknown = set(cfg) - _JSON_KEYS
    if unknown:
        raise ValueError(f"unknown config key(s) {sorted(unknown)}; "
                         f"allowed: {', '.join(sorted(_JSON_KEYS))}")
    return dict(cfg)


def render_from_json(cfg_path, out_path=None):
    """Render from a JSON file holding channel values and optional render options."""
    cfg = load_config(json.load(open(cfg_path)))
    if out_path:
        cfg["out_path"] = out_path
    return render_eeg_brain(**cfg)


def _self_check():
    cz, fp1, oz, t8 = (electrode_dir(n) for n in ("Cz", "Fp1", "Oz", "T8"))
    assert np.allclose(cz, [0, 0, 1], atol=1e-6), cz
    assert fp1[1] > 0.9 and fp1[0] < 0, fp1          # left frontal
    assert oz[1] < -0.9 and abs(oz[0]) < 1e-6, oz     # midline occipital
    assert t8[0] > 0.9, t8                            # right temporal

    e = np.array([cz, oz])
    # tight sigma: sampling at an electrode must recover ~its own value
    assert abs(scalp_interpolate([cz], e, [1.0, 0.0], sigma=0.15)[0] - 1.0) < 0.01
    assert abs(scalp_interpolate([oz], e, [1.0, 0.0], sigma=0.15)[0] - 0.0) < 0.01
    half = (cz + oz) / np.linalg.norm(cz + oz)        # equidistant from both
    mid = scalp_interpolate([half], e, [1.0, 0.0], sigma=1.0)[0]
    assert 0.4 < mid < 0.6, mid                       # blends in between

    t, lo, hi = normalize([5.0, 10.0, 15.0])
    assert (lo, hi) == (5.0, 15.0) and np.allclose(t, [0, 0.5, 1])
    assert np.allclose(normalize([7.0, 7.0])[0], 0.5)  # flat input, no divide-by-zero

    c = colormap([0.0, 1.0])
    assert c.shape == (2, 3) and c[1][0] > c[0][0] and c[0][2] > c[1][2]

    rwb = colormap([0.0, 0.5, 1.0], "red_white_blue")
    assert np.allclose(rwb[1], [1, 1, 1]), rwb[1]      # mid is white
    assert rwb[0][0] > 0.8 and rwb[0][2] < 0.2, rwb[0]  # low is red
    assert rwb[2][2] > 0.8 and rwb[2][0] < 0.2, rwb[2]  # high is blue

    custom = [(1.0, (0.0, 0.0, 1.0)), (0.0, (1.0, 0.0, 0.0))]  # deliberately unsorted
    cc = colormap([0.0, 1.0], custom)
    assert np.allclose(cc[0], [1, 0, 0]) and np.allclose(cc[1], [0, 0, 1]), cc
    for bad, why in [("nope", "unknown name"), ([(0.0, (1, 0, 0))], "one stop"),
                     ([(0.0, (1, 0)), (1.0, (0, 0, 1))], "2-component colour")]:
        try:
            resolve_cmap(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{why} should raise")
    try:
        electrode_dir("Nope")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown channel should raise")

    # framing: from any direction every vertex must land inside the frame, and the
    # binding axis must actually touch `fill` (otherwise it is framed too loosely).
    rng = np.random.default_rng(0)
    cloud = rng.normal(size=(400, 3)) * np.array([0.80, 1.05, 0.60])
    cloud[:, 2] = np.where(cloud[:, 2] < 0, cloud[:, 2] * 0.62, cloud[:, 2])
    centre = 0.5 * (cloud.min(axis=0) + cloud.max(axis=0))
    tan_h, tan_v = 18.0 / 62.0, (18.0 / 62.0) * 1000 / 1600
    for d in list(VIEWS.values()) + [(0, 0, -1), (0.3, -0.2, 0.05), (-1, 1, 1)]:
        basis = camera_basis(d)
        right, up, fwd = basis
        fill = 0.80
        dist = frame_distance(cloud, centre, basis, tan_h, tan_v, fill)
        rel = cloud - (centre + fwd * dist)
        depth = -(rel @ fwd)                       # distance in front of the camera
        assert depth.min() > 1e-6, f"{d}: geometry behind the camera"
        u = np.abs(rel @ right) / (depth * tan_h)  # 1.0 = frame edge
        w = np.abs(rel @ up) / (depth * tan_v)
        assert u.max() <= fill + 1e-9 and w.max() <= fill + 1e-9, (d, u.max(), w.max())
        assert max(u.max(), w.max()) > fill - 1e-6, (d, "framed looser than fill")
    for bad in ((0, 0, 0),):
        try:
            camera_basis(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("zero direction should raise")

    # video helpers (pure python/numpy — no Blender needed)
    assert _series_at({"Cz": [0.0, 10.0]}, 0.0)["Cz"] == 0.0
    assert _series_at({"Cz": [0.0, 10.0]}, 1.0)["Cz"] == 10.0
    assert _series_at({"Cz": [0.0, 10.0]}, 0.5)["Cz"] == 5.0
    assert _series_at({"Cz": 7.0}, 0.3)["Cz"] == 7.0  # scalar / single value holds

    top, right = VIEWS["top"], VIEWS["right_lateral"]
    assert np.allclose(_slerp_dir(top, right, 0.0), np.array(top) / np.linalg.norm(top))
    assert np.allclose(_slerp_dir(top, right, 1.0), np.array(right) / np.linalg.norm(right))
    mid = _slerp_dir(top, right, 0.5)
    assert abs(np.linalg.norm(mid) - 1.0) < 1e-9  # stays on the unit sphere
    ang_a = math.acos(np.clip(mid @ (np.array(top) / np.linalg.norm(top)), -1, 1))
    ang_b = math.acos(np.clip(mid @ (np.array(right) / np.linalg.norm(right)), -1, 1))
    assert abs(ang_a - ang_b) < 1e-6, (ang_a, ang_b)  # equidistant at the midpoint

    kfs = _resolve_view_keyframes([(0.0, "top"), (1.0, "right_lateral")])
    assert np.allclose(_direction_at(kfs, 0.0), kfs[0][1])
    assert np.allclose(_direction_at(kfs, 1.0), kfs[1][1])
    fixed = _resolve_view_keyframes("top")
    assert np.allclose(_direction_at(fixed, 0.7), fixed[0][1])  # single view never moves
    hold = _resolve_view_keyframes([(0.2, "top"), (0.8, "right_lateral")])
    assert np.allclose(_direction_at(hold, 0.0), hold[0][1])   # holds before first kf
    assert np.allclose(_direction_at(hold, 1.0), hold[1][1])   # holds after last kf
    try:
        _resolve_view_keyframes("nope")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown view name should raise")

    assert load_config({"Fp1": 1.0}) == {"channels": {"Fp1": 1.0}}  # bare map wraps
    assert load_config({"channels": {"Fp1": 1.0}, "view": "top"})["view"] == "top"
    try:
        load_config({"channels": {}, "viewpoint": "top"})
    except ValueError:
        pass
    else:
        raise AssertionError("unknown config key should raise")
    print("self-check ok")


def _cli():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    if "--self-check" in argv:
        _self_check()
        return
    out = argv[1] if len(argv) > 1 else "eeg_brain.png"
    if argv:
        render_from_json(argv[0], out)
    else:
        render_eeg_brain(demo_channels(), out_path=out)


if __name__ == "__main__":
    _cli()
