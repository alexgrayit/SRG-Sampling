#!/usr/bin/env python3
"""Make the x and v MP4s from saved SRG PNG frames. Requires ffmpeg on PATH."""

from pathlib import Path
import shutil
import subprocess


# Edit these settings, then run: python3 frames_to_mp4.py
# Relative paths are relative to this script. Absolute folder paths also work.
FRAME_FOLDER = "output/SRG_frames_afrx870_"

# Example names: SRG_<x or v>_<N in rounded millions>_<spring>_<damper>_<lin or nonlin>.mp4
# Set these to match your simulation; the frame count cannot determine N.
X_OUTPUT_NAME = "SRG_x_535_2_2_nonlin.mp4"
V_OUTPUT_NAME = "SRG_v_535_2_2_nonlin.mp4"
VIDEO_FPS = 60  # Same default as drag_listener.py.


def make_video(frame_directory, destination):
    frames = sorted(frame_directory.glob("frame_*.png"))
    if not frames:
        print(f"Skipping {frame_directory.name}: no frame_*.png files found.")
        return False

    print(f"Creating {destination.name} from {len(frames):,} saved frames...", flush=True)
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
            "-framerate", str(VIDEO_FPS),
            # Saved frames have zero-padded numbers, so glob order is frame order.
            # Unlike a numbered sequence, this also handles missing frame numbers.
            "-pattern_type", "glob", "-i", str(frame_directory / "frame_*.png"),
            "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            str(destination),
        ],
        check=True,
    )
    print(f"Saved {destination}", flush=True)
    return True


def main():
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg is required. Install it and ensure it is on PATH.")

    folder = Path(FRAME_FOLDER).expanduser()
    if not folder.is_absolute():
        folder = Path(__file__).resolve().parent / folder
    folder = folder.resolve()
    if not folder.is_dir():
        raise SystemExit(f"Frame folder does not exist: {folder}\nEdit FRAME_FOLDER first.")
    if VIDEO_FPS <= 0:
        raise SystemExit("VIDEO_FPS must be greater than zero.")

    # Save beside the frame folder, matching the existing output directory.
    # Existing MP4s with these names are overwritten; the PNGs are retained.
    created = 0
    for name, output_name in (("x", X_OUTPUT_NAME), ("v", V_OUTPUT_NAME)):
        created += make_video(folder / name, folder.parent / output_name)
    if not created:
        raise SystemExit("No videos created. Expected x/frame_*.png and v/frame_*.png.")


if __name__ == "__main__":
    try:
        main()
    except (OSError, subprocess.CalledProcessError) as error:
        raise SystemExit(f"Video creation failed: {error}") from error
