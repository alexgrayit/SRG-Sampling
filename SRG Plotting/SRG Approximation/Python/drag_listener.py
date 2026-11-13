import argparse
import numpy as np
import lcm
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from scipy.spatial import ConvexHull, QhullError
from collections import deque
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
from queue import Empty
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, VPacker
from matplotlib.text import Text

from lcm_messages.simulation_messages import SRGState
from lcm_messages.simulation_messages import SRGBatch
from lcm_messages.simulation_messages import startSim
from lcm_messages.simulation_messages import SimInfo
from lcm_messages.simulation_messages import SimDone

# Plotting preferences stay here. The run scripts optionally accept --mode.
PLOT_MODE = "hconvex"  # "raw", "convex", or "hconvex"
HULL_SHADE = True
HULL_ALPHA = 0.18
ARC_POINTS = 120
PLOT_INTERVAL = 0.1
SAVE_VIDEO = True  # Requires ffmpeg on PATH when enabled.
VIDEO_FPS = 30
VIDEO_DPI = 300
OUTPUT_DIRECTORY = Path(__file__).resolve().parent / "output"
plt.rcParams['figure.raise_window'] = False

k_d = None
k_p = None
linear_drag = None
model_ready = threading.Event()
simulation_done = threading.Event()
done_message = None
simulation_points_received = 0
# Recording has its own process; neither incoming data nor live plots wait for PNGs.
video_jobs = None
video_process = None
video_result = None
frames_directory = None
frame_counts = {"x": 0, "v": 0}
last_frame_received = {"x": -1, "v": -1}

# Raw mode intentionally retains every point, without a memory limit.
x_xs = deque()
x_ys = deque()
v_xs = deque()
v_ys = deque()
x_hull = np.empty(0, dtype=complex)
v_hull = np.empty(0, dtype=complex)
data_lock = threading.Lock()

plt.ion()
fig_x, ax_x = plt.subplots()
(line_x,) = ax_x.plot([], [], 'o', markersize=3)
fig_v, ax_v = plt.subplots()
(line_v,) = ax_v.plot([], [], 'o', markersize=3)

ax_x.grid(which='minor', linestyle=':', linewidth='0.5', color='gray', alpha=0.5)
ax_v.grid(which='minor', linestyle=':', linewidth='0.5', color='gray', alpha=0.5)
ax_x.set_title(r"SRG for $\langle x, u \rangle$")
ax_v.set_title(r"SRG for $\langle v, u \rangle$")

point_labels, title_boxes, original_tops, video_sizes = {}, {}, {}, {}
for name, fig, ax in (("x", fig_x, ax_x), ("v", fig_v, ax_v)):
    title = TextArea(ax.get_title(), textprops={"size": ax.title.get_fontsize()})
    # Received counts simulations; Boundary counts upper-hull vertices.
    counts = TextArea("Received: 0    Boundary: 0",
                      textprops={"size": ax.title.get_fontsize() - 3})
    box = AnchoredOffsetbox(loc="lower center", frameon=False, pad=0,
                            child=VPacker(children=[title, counts], align="center", pad=0, sep=3),
                            bbox_to_anchor=(0.5, 1.0), bbox_transform=ax.transAxes)
    ax.set_title("")
    ax.add_artist(box)
    point_labels[name], title_boxes[name] = counts, box
    original_tops[name] = fig.subplotpars.top
    video_sizes[name] = fig.get_size_inches().copy()
    fig.subplots_adjust(top=0.8)

# Separate upper/lower patches avoid filling the gap between the two hulls.
x_shades = [Polygon(np.empty((0, 2)), closed=True, color=line_x.get_color(), alpha=HULL_ALPHA) for _ in range(2)]
v_shades = [Polygon(np.empty((0, 2)), closed=True, color=line_v.get_color(), alpha=HULL_ALPHA) for _ in range(2)]
for ax, patches in ((ax_x, x_shades), (ax_v, v_shades)):
    for patch in patches:
        ax.add_patch(patch)
        patch.set_visible(False)

srg_x_xmin, srg_x_xmax = ax_x.get_xlim()
srg_x_ymin, srg_x_ymax = ax_x.get_ylim()
srg_v_xmin, srg_v_xmax = ax_v.get_xlim()
srg_v_ymin, srg_v_ymax = ax_v.get_ylim()


def coefficient_handler(channel, data):
    global k_d, k_p, linear_drag
    msg = SimInfo.decode(data)
    k_d, k_p, linear_drag = msg.k_d, msg.k_p, bool(msg.linear_drag)
    model_ready.set()


def done_handler(channel, data):
    global done_message
    done_message = SimDone.decode(data)
    simulation_done.set()


def plot_system_srg():
    global srg_x_xmin, srg_x_xmax, srg_x_ymin, srg_x_ymax
    global srg_v_xmin, srg_v_xmax, srg_v_ymin, srg_v_ymax
    # Always show the reference SRGs. For quadratic drag use the requested 2*k_d.
    A_x = np.array([[0, 1], [-k_p, -k_d]])
    B_x = np.array([[0], [1]])
    C_x = np.array([[1, 0]])
    D_x = np.array([[0]])
    SS_x = control.ss(A_x, B_x, C_x, D_x)
    A_v, B_v = A_x, B_x
    C_v = np.array([[0, 1]])
    D_v = D_x
    SS_v = control.ss(A_v, B_v, C_v, D_v)
    srg.srg_lti(control.tf(SS_x), shade=True, show_lower=True, ax=ax_x)
    srg.srg_lti(control.tf(SS_v), shade=True, show_lower=True, ax=ax_v)
    srg_x_xmin, srg_x_xmax = ax_x.get_xlim()
    srg_x_ymin, srg_x_ymax = ax_x.get_ylim()
    srg_v_xmin, srg_v_xmax = ax_v.get_xlim()
    srg_v_ymin, srg_v_ymax = ax_v.get_ylim()


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
    global x_hull, v_hull, simulation_points_received
    if SAVE_VIDEO and video_jobs is None:
        prepare_video_recording()
    msg = SRGBatch.decode(data)
    new_x = _srg_points(msg.L2_x, msg.x_norm, msg.u_norm)
    new_v = _srg_points(msg.L2_v, msg.v_norm, msg.u_norm)
    with data_lock:
        simulation_points_received += msg.n
        old_x, old_v = x_hull, v_hull
        # In raw mode these convex hulls are used only for the boundary counts.
        x_hull = _add_to_hull(x_hull, new_x)
        v_hull = _add_to_hull(v_hull, new_v)
        changed_x = _hull_changed(old_x, x_hull)
        changed_v = _hull_changed(old_v, v_hull)
        if PLOT_MODE == "raw":
            new_x = np.column_stack((new_x, new_x.conjugate())).ravel()
            new_v = np.column_stack((new_v, new_v.conjugate())).ravel()
            x_xs.extend(new_x.real)
            x_ys.extend(new_x.imag)
            v_xs.extend(new_v.real)
            v_ys.extend(new_v.imag)
        if SAVE_VIDEO:
            # Hull arrays are replaced, never modified, so each queued reference
            # retains that boundary. Raw mode sends only the new points.
            job = (simulation_points_received, x_hull, v_hull,
                   new_x if PLOT_MODE == "raw" else None,
                   new_v if PLOT_MODE == "raw" else None,
                   (changed_x, changed_v), False)
    if SAVE_VIDEO and (changed_x or changed_v or PLOT_MODE == "raw"):
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
    srg_handler(channel, msg.encode())


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


def update_plot_x(frame):
    global fig_x, ax_x, line_x, srg_x_xmin, srg_x_xmax, srg_x_ymin, srg_x_ymax
    with data_lock:
        x, y = _plot_data(x_xs, x_ys, x_hull, line_x, x_shades)
        point_labels["x"].set_text(f"Received: {simulation_points_received:,}    Boundary: {len(x_hull):,}")
    line_x.set_data(x, y)
    valid = np.isfinite(x) & np.isfinite(y)
    if np.any(valid):
        ax_x.set_xlim(min(min(x[valid]), srg_x_xmin) - 0.01, max(max(x[valid]), srg_x_xmax) + 0.01)
        ax_x.set_ylim(min(min(y[valid]), srg_x_ymin) - 0.01, max(max(y[valid]), srg_x_ymax) + 0.01)
    ax_x.set_aspect('equal')
    if plt.fignum_exists(fig_x.number):
        fig_x.canvas.draw_idle()
        fig_x.canvas.flush_events()
    return line_x,


def update_plot_v(frame):
    global fig_v, ax_v, line_v, srg_v_xmin, srg_v_xmax, srg_v_ymin, srg_v_ymax
    with data_lock:
        x, y = _plot_data(v_xs, v_ys, v_hull, line_v, v_shades)
        point_labels["v"].set_text(f"Received: {simulation_points_received:,}    Boundary: {len(v_hull):,}")
    line_v.set_data(x, y)
    valid = np.isfinite(x) & np.isfinite(y)
    if np.any(valid):
        ax_v.set_xlim(min(min(x[valid]), srg_v_xmin) - 0.01, max(max(x[valid]), srg_v_xmax) + 0.01)
        ax_v.set_ylim(min(min(y[valid]), srg_v_ymin) - 0.01, max(max(y[valid]), srg_v_ymax) + 0.01)
    ax_v.set_aspect('equal')
    if plt.fignum_exists(fig_v.number):
        fig_v.canvas.draw_idle()
        fig_v.canvas.flush_events()
    return line_v,


def save_video_frame(name, fig):
    global frames_directory
    if frames_directory is None:
        OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
        frames_directory = Path(tempfile.mkdtemp(prefix="SRG_frames_", dir=OUTPUT_DIRECTORY))
    directory = frames_directory / name
    directory.mkdir(exist_ok=True)
    size = fig.get_size_inches().copy()
    # All frames retain the initial dimensions, even if the live window is resized.
    fig.set_size_inches(video_sizes[name], forward=False)
    try:
        fig.savefig(directory / f"frame_{frame_counts[name]:08d}.png", dpi=VIDEO_DPI)
    finally:
        fig.set_size_inches(size, forward=False)
    frame_counts[name] += 1
    last_frame_received[name] = simulation_points_received


def prepare_video_recording():
    global video_jobs, frames_directory
    if video_jobs is not None:
        return
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    frames_directory = Path(tempfile.mkdtemp(prefix="SRG_frames_", dir=OUTPUT_DIRECTORY))
    # No frame dropping or queue capacity limit; recording can catch up later.
    video_jobs = multiprocessing.get_context("spawn").Queue()


def video_worker(jobs, result, options):
    global PLOT_MODE, VIDEO_DPI, ARC_POINTS, HULL_SHADE, k_p, k_d, linear_drag
    global frames_directory, video_sizes, simulation_points_received, x_hull, v_hull
    error = None
    try:
        # This process imports Matplotlib with Agg and owns separate figures.
        plt.rcParams.update(options["style"])
        plt.ioff()
        PLOT_MODE, VIDEO_DPI = options["mode"], options["dpi"]
        ARC_POINTS, HULL_SHADE = options["arc_points"], options["shade"]
        k_p, k_d, linear_drag = options["coefficients"]
        frames_directory, video_sizes = options["directory"], options["sizes"]
        for fig in (fig_x, fig_v):
            for text in fig.findobj(Text):
                text.set_usetex(plt.rcParams["text.usetex"])
                text.set_fontfamily(plt.rcParams["font.family"])
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
            simulation_points_received, x_hull, v_hull, new_x, new_v, changes, final = job
            if new_x is not None:
                x_xs.extend(new_x.real)
                x_ys.extend(new_x.imag)
                v_xs.extend(new_v.real)
                v_ys.extend(new_v.imag)
            for name, changed, fig, update in (("x", changes[0], fig_x, update_plot_x),
                                                ("v", changes[1], fig_v, update_plot_v)):
                if changed or (final and last_frame_received[name] != simulation_points_received):
                    update(None)
                    save_video_frame(name, fig)
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
               "shade": HULL_SHADE, "coefficients": (k_p, k_d, linear_drag),
               "directory": frames_directory, "sizes": video_sizes,
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
    video_jobs.put((simulation_points_received, x_hull, v_hull, None, None, (False, False), True))
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
    dynamics = "lin" if linear_drag else "nonlin"
    return f"SRG_{name}_{millions}_{spring}_{damper}_{dynamics}"


def save_outputs():
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    status = "cancelled" if done_message.cancelled else "finished"
    print(f"C++ {status}: {simulation_points_received:,} simulations received.", flush=True)
    if simulation_points_received != done_message.n_simulations:
        print(f"C++ reported {done_message.n_simulations:,} published simulations; this differs from the received count.", flush=True)
    for name, fig in (("x", fig_x), ("v", fig_v)):
        stem = output_stem(name)
        title_boxes[name].set_visible(False)
        top = fig.subplotpars.top
        fig.subplots_adjust(top=original_tops[name])
        try:
            fig.savefig(OUTPUT_DIRECTORY / f"{stem}.pdf", bbox_inches="tight")
        finally:
            title_boxes[name].set_visible(True)
            fig.subplots_adjust(top=top)
        print(f"Saved {stem}.pdf", flush=True)
    if SAVE_VIDEO:
        finish_video_recording()
        for name in ("x", "v"):
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


LCM_URL = "udpm://239.255.76.67:7667?ttl=0&recv_buf_size=8388608"


def lcm_thread():
    lc = lcm.LCM(LCM_URL)
    lc.subscribe("SimInfo", coefficient_handler)
    subscription = lc.subscribe("SRGBatch", srg_handler)
    subscription.set_queue_capacity(0)
    lc.subscribe("SRGState", _srg_state_handler)
    lc.subscribe("SimDone", done_handler)
    # All subscriptions exist before readiness; C++ replies once with SimInfo.
    startMsg = startSim()
    startMsg.sim_name = "drag_sim"
    lc.publish("startSim", startMsg.encode())
    try:
        while not simulation_done.is_set():
            lc.handle()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("raw", "convex", "hconvex"), default=PLOT_MODE)
    PLOT_MODE = parser.parse_args().mode
    if SAVE_VIDEO and shutil.which("ffmpeg") is None:
        raise RuntimeError("Video output requires ffmpeg on PATH.")
    if SAVE_VIDEO:
        prepare_video_recording()
    t = threading.Thread(target=lcm_thread, daemon=True)
    t.start()
    reference_plotted = False
    outputs_saved = False
    while not outputs_saved or plt.fignum_exists(fig_x.number) or plt.fignum_exists(fig_v.number):
        if model_ready.is_set() and not reference_plotted:
            plot_system_srg()
            reference_plotted = True
            if SAVE_VIDEO:
                start_video_recorder()
        update_plot_x(None)
        update_plot_v(None)
        if simulation_done.is_set() and reference_plotted and not outputs_saved:
            # The completion callback follows the final batch on the LCM thread.
            update_plot_x(None)
            update_plot_v(None)
            save_outputs()
            outputs_saved = True
        # Process GUI events without the repeated show() call in plt.pause().
        if plt.fignum_exists(fig_x.number):
            fig_x.canvas.start_event_loop(PLOT_INTERVAL)
        elif plt.fignum_exists(fig_v.number):
            fig_v.canvas.start_event_loop(PLOT_INTERVAL)
        else:
            time.sleep(PLOT_INTERVAL)
