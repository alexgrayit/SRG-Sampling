import argparse
import numpy as np
import lcm
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from scipy.spatial import ConvexHull, QhullError
from collections import deque
from copy import copy
import plotting
import control
import LTI_SRG as srg
import threading
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import multiprocessing
import os
import traceback
from queue import Empty, SimpleQueue
from matplotlib.backends.registry import backend_registry
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, VPacker
from matplotlib.texmanager import TexManager
from matplotlib.text import Text
from matplotlib.ticker import ScalarFormatter

from lcm_messages.simulation_messages import SRGState
from lcm_messages.simulation_messages import SRGBatch
from lcm_messages.simulation_messages import startSim
from lcm_messages.simulation_messages import SimInfo
from lcm_messages.simulation_messages import SimDone
from lcm_messages.simulation_messages import SimPlotInfo
from lcm_messages.simulation_messages import SRGPlotBatch
from lcm_messages.simulation_messages import SimFrictionInfo

# Plotting preferences stay here. The run scripts optionally accept --mode.
PLOT_MODE = "hconvex"  # "raw", "convex", or "hconvex"
HULL_SHADE = True
HULL_ALPHA = 0.18
HULL_COLOR = "tab:purple"
ARC_POINTS = 120
PLOT_INTERVAL = 0.1
SAVE_VIDEO = True  # Requires ffmpeg on PATH when enabled.
VIDEO_FPS = 30
VIDEO_DPI = 300
OUTPUT_DIRECTORY = Path(__file__).resolve().parent / "output"
OUTPUT_LABELS = {"x": "x", "v": "v", "vrx": "v+rx"}
plt.rcParams['figure.raise_window'] = False

k_d = None
k_p = None
r = None
linear_drag = None
ideal_only = False
friction_enabled = False
model_ready = threading.Event()
simulation_done = threading.Event()
lcm_errors = SimpleQueue()
done_message = None
simulation_points_received = 0
# Recording has its own process; neither incoming data nor live plots wait for PNGs.
video_jobs = None
video_process = None
video_result = None
frames_directory = None
frame_counts = dict.fromkeys(OUTPUT_LABELS, 0)
last_frame_received = dict.fromkeys(OUTPUT_LABELS, -1)
data_lock = threading.Lock()

if __name__ == "__main__" and plt.rcParams["text.usetex"]:
    try:
        TexManager.make_dvi(r"$x$", plt.rcParams["font.size"])
    except (RuntimeError, OSError):
        print("LaTeX could not render the plot labels. Using Matplotlib's math text.", flush=True)
        plt.rcParams.update({"text.usetex": False, "font.family": "serif", "mathtext.fontset": "cm"})

class SRGPlot:
    def __init__(self):
        self.hull = np.empty(0, dtype=complex)
        # Raw mode intentionally retains every point.
        self.raw_x, self.raw_y = deque(), deque()

    def create_figure(self, label):
        self.fig, self.ax = plt.subplots()
        (self.line,) = self.ax.plot([], [], 'o', markersize=3, color=HULL_COLOR)
        self.ax.grid(which='minor', linestyle=':', linewidth='0.5', color='gray', alpha=0.5)
        font_size = self.ax.title.get_fontsize()
        title = TextArea(rf"SRG for $\langle {label}, u \rangle$", textprops={"size": font_size})
        self.counts = TextArea("Received: 0    Boundary: 0", textprops={"size": font_size - 3})
        children = [title] if ideal_only else [title, self.counts]
        self.title_box = AnchoredOffsetbox(
            loc="lower center", frameon=False, pad=0,
            child=VPacker(children=children, align="center", pad=0, sep=3),
            bbox_to_anchor=(0.5, 1.0), bbox_transform=self.ax.transAxes,
        )
        self.ax.add_artist(self.title_box)
        self.original_top = self.fig.subplotpars.top
        self.video_size = self.fig.get_size_inches().copy()
        self.fig.subplots_adjust(top=0.8)

        # Fill each half of the sampled hull independently.
        self.shades = []
        for _ in range(2):
            patch = Polygon(np.empty((0, 2)), closed=True, color=HULL_COLOR, alpha=HULL_ALPHA)
            self.ax.add_patch(patch)
            patch.set_visible(False)
            self.shades.append(patch)


plots = {name: SRGPlot() for name in OUTPUT_LABELS}


def plot_names():
    # Older simulators send position and velocity data only.
    return ("x", "v") if r is None else tuple(OUTPUT_LABELS)


def coefficient_handler(channel, data):
    global k_d, k_p, r, linear_drag, ideal_only, friction_enabled
    if model_ready.is_set():
        return
    message_types = {"SimInfo": SimInfo, "SimPlotInfo": SimPlotInfo, "SimFrictionInfo": SimFrictionInfo}
    msg = message_types[channel].decode(data)
    k_d, k_p, linear_drag = msg.k_d, msg.k_p, bool(msg.linear_drag)
    r = getattr(msg, "r", None)
    ideal_only = bool(getattr(msg, "ideal_only", False))
    friction_enabled = bool(getattr(msg, "friction_enabled", False))
    if ideal_only:
        print("Plotting the analytical linear SRGs.", flush=True)
    else:
        print("Connected to the simulator. Waiting for SRG results.", flush=True)
    model_ready.set()


def done_handler(channel, data):
    global done_message
    done_message = SimDone.decode(data)
    simulation_done.set()


def plot_system_srg():
    denominator = [1, k_d, k_p]
    numerators = {"x": [1], "v": [1, 0], "vrx": [1, r]}
    for name in plot_names():
        plot = plots[name]
        plot.create_figure(OUTPUT_LABELS[name])
        numerator = numerators[name]
        transfer = control.tf(numerator, denominator)
        omega = None
        if name != "x" and k_p > 0:
            # Include DC and the positive real-axis crossing, if one exists.
            omega = np.r_[0.0, srg._default_omega(transfer, n=5000)]
            crossing = k_p - numerator[-1] * k_d
            if crossing > 0:
                omega = np.r_[omega, np.sqrt(crossing)]
        n_lines, n_patches = len(plot.ax.lines), len(plot.ax.patches)
        srg.srg_lti(transfer, omega=omega, shade=name != "v", show_lower=True, ax=plot.ax)
        plot.ideal_artists = [*plot.ax.lines[n_lines:], *plot.ax.patches[n_patches:]]
        plot.ideal_xlim = plot.ax.get_xlim()
        plot.ideal_ylim = plot.ax.get_ylim()


def _convex_hull_complex(points):
    points = np.asarray(points, dtype=complex)
    xy = np.unique(np.column_stack((points.real, points.imag)), axis=0)
    if len(xy) < 3:
        return xy[:, 0] + 1j * xy[:, 1]
    try:
        xy = xy[ConvexHull(xy).vertices]
    except QhullError:
        # Monotone chain also handles collinear points, retaining only endpoints.
        # https://en.wikipedia.org/wiki/Convex_hull_algorithms
        def cross(a, b, c):
            ab = b.astype(np.longdouble) - a
            ac = c.astype(np.longdouble) - a
            return ab[0] * ac[1] - ab[1] * ac[0]

        lower, upper = [], []
        for p in xy:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
                lower.pop()
            lower.append(p)
        for p in xy[::-1]:
            while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
                upper.pop()
            upper.append(p)
        xy = np.asarray(lower[:-1] + upper[:-1])
    return xy[:, 0] + 1j * xy[:, 1]


def _add_to_hull(hull, points):
    # Only the upper half plane (including the real axis) enters either hull.
    points = points[np.isfinite(points) & (points.imag >= 0)]
    if PLOT_MODE == "hconvex":
        # H-geodesics become straight chords in the Beltrami-Klein disk.
        # https://arxiv.org/abs/2106.05650 (Section IV)
        # https://en.wikipedia.org/wiki/Beltrami%E2%80%93Klein_model
        points = srg.beltrami_klein(points)
    # conv(conv(old) union new) = conv(old union new), so discard interior points.
    # https://en.wikipedia.org/wiki/Convex_hull
    return _convex_hull_complex(np.concatenate((hull, points)))


def _srg_points(L2, norm, u_norm):
    L2, norm, u_norm = (np.asarray(a, dtype=float) for a in (L2, norm, u_norm))
    valid = np.isfinite(L2) & np.isfinite(norm) & np.isfinite(u_norm) & (u_norm > 0) & (norm >= 0)
    L2, norm, u_norm = L2[valid], norm[valid], u_norm[valid]
    # z = (||y||/||u||)*exp(i*theta), cos(theta) = <y,u>/(||y||*||u||).
    cosine = np.divide(L2, norm * u_norm, out=np.zeros_like(L2), where=norm > 0)
    cosine = np.clip(cosine, -1.0, 1.0)
    gain = norm / u_norm
    # sqrt(1-cos(theta)^2) keeps real-axis points exactly on that axis.
    points = gain * (cosine + 1j * np.sqrt(np.maximum(0, 1 - cosine**2)))
    return points[np.isfinite(points)]


def srg_handler(channel, data):
    batch_type = SRGPlotBatch if channel == "SRGPlotBatch" else SRGBatch
    _process_batch(batch_type.decode(data), combined=batch_type is SRGPlotBatch)


def _process_batch(msg, combined=False):
    global simulation_points_received
    if ideal_only:
        return
    if SAVE_VIDEO and video_jobs is None:
        prepare_video_recording()
    names = OUTPUT_LABELS if combined else ("x", "v")
    points = {
        name: _srg_points(getattr(msg, f"L2_{name}"), getattr(msg, f"{name}_norm"), msg.u_norm)
        for name in names
    }
    changed = set()
    raw_points = {}
    with data_lock:
        simulation_points_received += msg.n
        for name, new_points in points.items():
            plot = plots[name]
            old_hull = plot.hull
            # Raw mode also tracks the hull for the boundary count.
            plot.hull = _add_to_hull(old_hull, new_points)
            if _hull_changed(old_hull, plot.hull):
                changed.add(name)
            if PLOT_MODE == "raw":
                mirrored = np.column_stack((new_points, new_points.conjugate())).ravel()
                plot.raw_x.extend(mirrored.real)
                plot.raw_y.extend(mirrored.imag)
                raw_points[name] = mirrored
        if SAVE_VIDEO and (changed or raw_points):
            # Hull arrays are replaced, never modified, so each queued reference
            # retains that boundary. Raw mode sends only the new points.
            hulls = {name: plots[name].hull for name in names}
            job = (simulation_points_received, hulls, raw_points, changed, False)
        else:
            job = None
    if job is not None:
        video_jobs.put(job)


def _hull_changed(old, new):
    # The same cyclic hull can start at a different vertex.
    return len(old) != len(new) or not np.array_equal(np.sort_complex(old), np.sort_complex(new))


def _srg_state_handler(channel, data):
    # Retain compatibility with a single SRGState message.
    state = SRGState.decode(data)
    msg = SRGBatch()
    msg.n = 1
    for name in ("x_norm", "v_norm", "u_norm", "L2_x", "L2_v"):
        setattr(msg, name, [getattr(state, name)])
    _process_batch(msg)


def _hull_path(hull):
    if len(hull) == 0:
        return hull
    if PLOT_MODE == "convex":
        return np.append(hull, hull[0]) if len(hull) >= 3 else hull
    # Invert sampled disk chords to draw upper circular/vertical H-geodesics.
    vertices = srg.beltrami_klein_inverse(hull)
    if len(hull) == 1:
        return vertices
    edges = len(hull) if len(hull) >= 3 else 1
    parts = []
    for i in range(edges):
        j = (i + 1) % len(hull)
        chord = hull[i] + np.linspace(0, 1, ARC_POINTS) * (hull[j] - hull[i])
        part = srg.beltrami_klein_inverse(chord)
        part[0], part[-1] = vertices[i], vertices[j]
        parts.append(part if i == 0 else part[1:])
    return np.concatenate(parts)


def _plot_data(xs, ys, hull, line, patches):
    if PLOT_MODE == "raw":
        line.set_linestyle('None')
        line.set_marker('o')
        for patch in patches:
            patch.set_visible(False)
        return np.asarray(xs), np.asarray(ys)
    upper = _hull_path(hull)
    lower = upper.conjugate()
    line.set_linestyle('-')
    line.set_marker('o' if len(hull) == 1 else 'None')
    for patch, boundary in zip(patches, (upper, lower)):
        patch.set_xy(np.column_stack((boundary.real, boundary.imag)))
        patch.set_visible(HULL_SHADE and len(hull) >= 3)
    # Mirror the completed upper hull; never hull both half planes together.
    points = np.concatenate((upper, [complex(np.nan, np.nan)], lower)) if len(upper) else upper
    return points.real, points.imag


def update_plot(plot):
    with data_lock:
        x, y = _plot_data(plot.raw_x, plot.raw_y, plot.hull, plot.line, plot.shades)
        plot.counts.set_text(f"Received: {simulation_points_received:,}    Boundary: {len(plot.hull):,}")
    plot.line.set_data(x, y)
    valid = np.isfinite(x) & np.isfinite(y)
    if np.any(valid):
        xmin, xmax = plot.ideal_xlim
        ymin, ymax = plot.ideal_ylim
        plot.ax.set_xlim(min(min(x[valid]), xmin) - 0.01, max(max(x[valid]), xmax) + 0.01)
        plot.ax.set_ylim(min(min(y[valid]), ymin) - 0.01, max(max(y[valid]), ymax) + 0.01)
    plot.ax.set_aspect('equal')
    if plt.fignum_exists(plot.fig.number):
        plot.fig.canvas.draw_idle()
        plot.fig.canvas.flush_events()


def save_video_frame(name, plot):
    directory = frames_directory / name
    directory.mkdir(exist_ok=True)
    fig = plot.fig
    size = fig.get_size_inches().copy()
    # All frames retain the initial dimensions, even if the live window is resized.
    fig.set_size_inches(plot.video_size, forward=False)
    try:
        fig.savefig(directory / f"frame_{frame_counts[name]:08d}.png", dpi=VIDEO_DPI)
    finally:
        fig.set_size_inches(size, forward=False)
    frame_counts[name] += 1
    last_frame_received[name] = simulation_points_received


def prepare_video_recording():
    global video_jobs, frames_directory
    with data_lock:
        if video_jobs is None:
            OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
            frames_directory = Path(tempfile.mkdtemp(prefix="SRG_frames_", dir=OUTPUT_DIRECTORY))
            # No frame dropping or queue capacity limit.
            video_jobs = multiprocessing.get_context("spawn").Queue()


def video_worker(jobs, result, options):
    global PLOT_MODE, VIDEO_DPI, ARC_POINTS, HULL_SHADE, k_p, k_d, r
    global frames_directory, simulation_points_received
    error = None
    try:
        # This process imports Matplotlib with Agg and owns separate figures.
        plt.rcParams.update(options["style"])
        plt.ioff()
        PLOT_MODE, VIDEO_DPI = options["mode"], options["dpi"]
        ARC_POINTS, HULL_SHADE = options["arc_points"], options["shade"]
        k_p, k_d, r = options["coefficients"]
        frames_directory = options["directory"]
        plot_system_srg()
        plt.close("all")  # Keep the figure objects, with no live GUI/event handling.
        parent = multiprocessing.parent_process()
        while True:
            if os.getppid() != parent.pid or not parent.is_alive():
                break
            try:
                job = jobs.get(timeout=0.25)
            except Empty:
                continue
            if job is None:
                break
            simulation_points_received, hulls, raw_points, changed, final = job
            for name, hull in hulls.items():
                plot = plots[name]
                plot.hull = hull
                if name in raw_points:
                    plot.raw_x.extend(raw_points[name].real)
                    plot.raw_y.extend(raw_points[name].imag)
                if name in changed or (final and last_frame_received[name] != simulation_points_received):
                    update_plot(plot)
                    save_video_frame(name, plot)
    except Exception:
        error = traceback.format_exc()
    finally:
        try:
            result.send((frame_counts, last_frame_received, error))
        except (BrokenPipeError, OSError):
            pass  # The live plot process was closed before recording completed.
        result.close()


def start_video_recorder():
    global video_process, video_result
    if video_process is not None:
        return
    prepare_video_recording()
    context = multiprocessing.get_context("spawn")
    video_result, sender = context.Pipe(duplex=False)
    options = {"mode": PLOT_MODE, "dpi": VIDEO_DPI, "arc_points": ARC_POINTS,
               "shade": HULL_SHADE, "coefficients": (k_p, k_d, r),
               "directory": frames_directory,
               "style": {key: value for key, value in plt.rcParams.items()
                         if key not in ("backend", "interactive")}}
    video_process = context.Process(target=video_worker, args=(video_jobs, sender, options), daemon=True)
    # Spawn must select a non-GUI backend before importing this script in the child.
    previous_backend = os.environ.get("MPLBACKEND")
    os.environ["MPLBACKEND"] = "Agg"
    try:
        video_process.start()
    finally:
        sender.close()
        if previous_backend is None:
            os.environ.pop("MPLBACKEND", None)
        else:
            os.environ["MPLBACKEND"] = previous_backend


def finish_video_recording():
    global frame_counts, last_frame_received
    start_video_recorder()
    # Only completion waits for the recorder, after the live data stream has ended.
    hulls = {name: plots[name].hull for name in plot_names()}
    video_jobs.put((simulation_points_received, hulls, {}, set(), True))
    video_jobs.put(None)
    print("Finishing queued video frames...", flush=True)
    try:
        frame_counts, last_frame_received, error = video_result.recv()
    except EOFError:
        error = "The video recorder exited without returning its frame counts."
    video_process.join()
    video_result.close()
    video_jobs.close()
    if error or video_process.exitcode != 0:
        video_jobs.cancel_join_thread()
        raise RuntimeError(error or f"Video recorder exited with status {video_process.exitcode}.")
    video_jobs.join_thread()


def output_stem(name):
    # Round halves up; N counts received simulations in millions.
    millions = (simulation_points_received + 500_000) // 1_000_000
    spring = int(np.floor(k_p + 0.5))
    damper = int(np.floor(k_d + 0.5))
    dynamics = "lin" if ideal_only or (linear_drag and not friction_enabled) else "nonlin"
    stem = f"SRG_{name}_{millions}_{spring}_{damper}_{dynamics}"
    if name == "vrx":
        stem += f"_r{r:.12g}"
    if friction_enabled:
        stem += "_friction"
    return stem + "_ideal_only" if ideal_only else stem


def save_plot_files(fig, stem):
    pdf = OUTPUT_DIRECTORY / f"{stem}.pdf"
    svg = OUTPUT_DIRECTORY / f"{stem}.svg"
    fig.savefig(pdf, bbox_inches="tight")
    print(f"Saved {pdf.name}", flush=True)

    # Use a bundled font for SVG layout and keep the labels as LaTeX source
    # in SVG <text> elements for Inkscape's PDF+LaTeX export.
    svg_font = "DejaVu Serif"
    labels = fig.findobj(Text)
    text_settings = [
        (label, label.get_usetex(), label.get_parse_math(), label.get_fontfamily())
        for label in labels
    ]
    numeric_axes = [
        (axis, formatter)
        for ax in fig.axes
        for axis in (ax.xaxis, ax.yaxis)
        if isinstance(formatter := axis.get_major_formatter(), ScalarFormatter)
    ]
    try:
        with plt.rc_context({
            "text.usetex": False,
            "text.parse_math": False,
            "svg.fonttype": "none",
            "font.family": svg_font,
        }):
            # Measure the tick values without counting LaTeX commands as text.
            for axis, formatter in numeric_axes:
                plain = copy(formatter)
                if hasattr(plain, "set_usetex"):
                    plain.set_usetex(False)
                else:
                    plain._usetex = False  # Matplotlib 3.9 has no public setter.
                plain.set_useMathText(False)
                axis.set_major_formatter(plain)
            for label in labels:
                label.set_usetex(False)
                label.set_parse_math(False)
                label.set_fontfamily(svg_font)
            fig.savefig(svg, bbox_inches="tight")
        # Normalize any remaining math labels for Inkscape's LaTeX export.
        source = svg.read_text(encoding="utf-8")
        source = source.replace(r"\mathdefault", "").replace("\N{MINUS SIGN}", "-")
        svg.write_text(source, encoding="utf-8")
    finally:
        for axis, formatter in numeric_axes:
            axis.set_major_formatter(formatter)
        for label, use_tex, parse_math, family in text_settings:
            label.set_usetex(use_tex)
            label.set_parse_math(parse_math)
            label.set_fontfamily(family)
    print(f"Saved {svg.name}", flush=True)


def add_axis_tip_labels(ax):
    # Use point offsets from the zero-centred spines at their outer tips.
    x_label = ax.xaxis.label
    y_label = ax.yaxis.label
    y_offset = ax.yaxis.get_offset_text()

    def imag_anchor(renderer):
        # Leave room for the scale factor above the spine in scientific notation.
        if y_offset.get_visible() and y_offset.get_text():
            return y_offset.get_window_extent(renderer)
        return ax.bbox

    real = ax.annotate(
        x_label.get_text(), (1, 0), xycoords=("axes fraction", "data"),
        xytext=(5, 0), textcoords="offset points", ha="left", va="center",
        fontproperties=x_label.get_fontproperties(), color=x_label.get_color(),
        annotation_clip=False,
    )
    imag = ax.annotate(
        y_label.get_text(), (0, 1), xycoords=("data", imag_anchor),
        xytext=(0, 5), textcoords="offset points", ha="center", va="bottom",
        fontproperties=y_label.get_fontproperties(), color=y_label.get_color(),
        annotation_clip=False,
    )
    return real, imag


def save_outputs():
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    if not ideal_only:
        status = "cancelled" if done_message.cancelled else "finished"
        print(f"C++ {status}: {simulation_points_received:,} simulations received.", flush=True)
        if simulation_points_received != done_message.n_simulations:
            print(f"C++ reported {done_message.n_simulations:,} published simulations; this differs from the received count.", flush=True)
    for name in plot_names():
        plot = plots[name]
        fig = plot.fig
        stem = output_stem(name)
        plot.title_box.set_visible(False)
        top = fig.subplotpars.top
        fig.subplots_adjust(top=plot.original_top)
        axis_labels = (plot.ax.xaxis.label, plot.ax.yaxis.label)
        axis_label_visibility = [label.get_visible() for label in axis_labels]
        tip_labels = add_axis_tip_labels(plot.ax)
        for label in axis_labels:
            label.set_visible(False)
        ideal_visibility = [artist.get_visible() for artist in plot.ideal_artists]
        try:
            save_plot_files(fig, stem)
            if not ideal_only:
                for artist in plot.ideal_artists:
                    artist.set_visible(False)
                save_plot_files(fig, f"{stem}_no_ideal")
        finally:
            for artist, visible in zip(plot.ideal_artists, ideal_visibility):
                artist.set_visible(visible)
            for label in tip_labels:
                label.remove()
            for label, visible in zip(axis_labels, axis_label_visibility):
                label.set_visible(visible)
            plot.title_box.set_visible(True)
            fig.subplots_adjust(top=top)
    if SAVE_VIDEO and not ideal_only:
        finish_video_recording()
        for name in plot_names():
            destination = OUTPUT_DIRECTORY / f"{output_stem(name)}.mp4"
            print(f"Creating {destination.name} from {frame_counts[name]:,} frames...", flush=True)
            subprocess.run([
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-framerate", str(VIDEO_FPS), "-start_number", "0",
                "-i", str(frames_directory / name / "frame_%08d.png"),
                "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                str(destination),
            ], check=True)
            print(f"Saved {destination.name}", flush=True)
        print(f"Saved frames in {frames_directory}", flush=True)


LCM_URL = os.environ.get("SRG_LCM_URL") or "udpm://239.255.76.67:7667?ttl=0&recv_buf_size=8388608"


def lcm_thread():
    try:
        lc = lcm.LCM(LCM_URL)
        lc.subscribe("SimInfo", coefficient_handler)
        lc.subscribe("SimPlotInfo", coefficient_handler)
        lc.subscribe("SimFrictionInfo", coefficient_handler)
        for channel in ("SRGBatch", "SRGPlotBatch"):
            lc.subscribe(channel, srg_handler).set_queue_capacity(0)
        lc.subscribe("SRGState", _srg_state_handler)
        lc.subscribe("SimDone", done_handler)
        start_message = startSim()
        start_message.sim_name = "drag_sim"
        ready = start_message.encode()
        simulator_pid = int(os.environ.get("SRG_SIM_PID", "0"))
        deadline = time.monotonic() + 60
        next_request = 0
        print("Plotter listening on LCM.", flush=True)
        while not simulation_done.is_set():
            now = time.monotonic()
            if not model_ready.is_set() and now >= next_request:
                if now >= deadline:
                    raise RuntimeError("No simulator responded over LCM within 60 seconds.")
                lc.publish("startSim", ready)
                next_request = now + 1
            handled = lc.handle_timeout(250)
            if handled < 0:
                raise RuntimeError("LCM stopped receiving messages.")
            if handled == 0 and simulator_pid:
                try:
                    os.kill(simulator_pid, 0)
                except ProcessLookupError as error:
                    raise RuntimeError("The simulator exited before sending its completion message.") from error
    except Exception as error:
        lcm_errors.put(error)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("raw", "convex", "hconvex"), default=PLOT_MODE)
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()
    PLOT_MODE = args.mode
    SAVE_VIDEO = SAVE_VIDEO and not args.no_video
    backend, gui_framework = backend_registry.resolve_backend(plt.get_backend())
    if gui_framework is None:
        print(f"Matplotlib is using {backend}. Results will be saved to files without live windows.", flush=True)
    plt.ion()
    t = threading.Thread(target=lcm_thread, daemon=True)
    t.start()
    reference_plotted = False
    outputs_saved = False
    while True:
        if not lcm_errors.empty():
            raise RuntimeError("The plotter could not receive simulation data over LCM.") from lcm_errors.get_nowait()
        if simulation_done.is_set() and not model_ready.is_set():
            raise RuntimeError("The plotter received completion before receiving the simulation settings.")
        if model_ready.is_set() and not reference_plotted:
            plot_system_srg()
            reference_plotted = True
            if SAVE_VIDEO and not ideal_only:
                if shutil.which("ffmpeg") is None:
                    raise RuntimeError("Video output requires ffmpeg on PATH.")
                start_video_recorder()
        if reference_plotted:
            done = simulation_done.is_set()
            for name in plot_names():
                update_plot(plots[name])
            if done and not outputs_saved:
                save_outputs()
                outputs_saved = True
            live_figure = next((plots[name].fig for name in plot_names()
                                if plt.fignum_exists(plots[name].fig.number)), None)
            if outputs_saved and (gui_framework is None or live_figure is None):
                break
            if live_figure is not None and gui_framework is not None:
                # Process GUI events without the repeated show() in plt.pause().
                live_figure.canvas.start_event_loop(PLOT_INTERVAL)
                continue
        time.sleep(PLOT_INTERVAL)
