import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull, QhullError
import control as ct
import plotting


# =========================
# Geometry / mappings
# =========================

def beltrami_klein(z):
    """Map upper half-plane -> Beltrami-Klein disk."""
    z = np.asarray(z, dtype=np.complex128)
    g = (z - 1j) / (z + 1j)
    return 2 * g / (1 + np.abs(g) ** 2)


def beltrami_klein_inverse(w):
    """Inverse Beltrami-Klein map."""
    w = np.asarray(w, dtype=np.complex128)

    rad = 1 - w * np.conj(w)
    rad = np.real(rad)
    rad = np.clip(rad, 0.0, None)

    return (np.imag(w) - 1j * np.sqrt(rad)) / (np.real(w) - 1)


def arc_min(z1, z2, n=120):
    """
    Hyperbolic geodesic in the upper half-plane between z1 and z2.
    Returns dense samples along the arc.
    """
    z1 = complex(z1)
    z2 = complex(z2)

    x1, y1 = np.real(z1), np.imag(z1)
    x2, y2 = np.real(z2), np.imag(z2)

    if np.isclose(z1, z2):
        return np.array([z1], dtype=np.complex128)

    if np.isclose(x1, x2):
        # Vertical geodesic
        return x1 + 1j * np.linspace(y1, y2, n)

    xc = ((y1 - y2) * (y1 + y2)) / (2 * (x1 - x2)) + (x1 + x2) / 2
    r = np.sqrt((x1 - xc) ** 2 + y1 ** 2)

    theta1 = np.angle(z1 - xc)
    theta2 = np.angle(z2 - xc)

    theta = np.linspace(theta1, theta2, n)
    return xc + r * np.exp(1j * theta)


def _convex_hull_complex(points):
    """Return convex hull vertices in cyclic order."""
    points = np.asarray(points, dtype=np.complex128)
    if points.size == 0:
        return points

    xy = np.column_stack([np.real(points), np.imag(points)])
    xy = np.unique(np.round(xy, decimals=12), axis=0)

    if len(xy) < 3:
        return xy[:, 0] + 1j * xy[:, 1]

    try:
        hull = ConvexHull(xy)
        verts = xy[hull.vertices]
        return verts[:, 0] + 1j * verts[:, 1]
    except QhullError:
        return xy[:, 0] + 1j * xy[:, 1]


def _densify_closed_arc_chain(vertices, arc_points=120):
    """
    Given closed polygon vertices [v0, v1, ..., vk, v0], replace each segment
    by the upper-half-plane geodesic arc.
    """
    dense = []
    for k in range(len(vertices) - 1):
        seg = arc_min(vertices[k], vertices[k + 1], n=arc_points)
        if k > 0:
            seg = seg[1:]  # avoid duplicating shared endpoint
        dense.append(seg)
    return np.concatenate(dense)


def _make_symmetric_polygon(upper_boundary):
    """
    Construct a filled SRG polygon in the complex plane by reflecting the upper
    boundary across the real axis.
    """
    upper_boundary = np.asarray(upper_boundary, dtype=np.complex128)

    if len(upper_boundary) < 2:
        return upper_boundary

    lower_boundary = np.conj(upper_boundary[::-1])

    # Avoid duplicating first/last points if they sit on the real axis
    poly = np.concatenate([upper_boundary, lower_boundary[1:-1]])
    return poly


# =========================
# Frequency response helpers
# =========================

def _default_omega(sys, n=2000):
    """Choose a reasonable frequency grid from poles/zeros."""
    try:
        p = np.asarray(ct.poles(sys), dtype=np.complex128)
    except Exception:
        p = np.array([], dtype=np.complex128)

    try:
        z = np.asarray(ct.zeros(sys), dtype=np.complex128)
    except Exception:
        z = np.array([], dtype=np.complex128)

    wz = np.abs(np.concatenate([p, z]))
    wz = wz[np.isfinite(wz)]
    wz = wz[wz > 0]

    if wz.size == 0:
        return np.logspace(-3, 3, n)

    wmin = 10 ** (np.floor(np.log10(np.min(wz))) - 2)
    wmax = 10 ** (np.ceil(np.log10(np.max(wz))) + 2)
    return np.logspace(np.log10(wmin), np.log10(wmax), n)


def _upper_nyquist(sys, omega):
    """
    Compute positive-frequency response and fold into upper half-plane using
    abs(imag), matching the MATLAB/Julia logic.
    """
    mag, phase, omega = ct.frequency_response(sys, omega=omega)
    mag = np.squeeze(np.asarray(mag))
    phase = np.squeeze(np.asarray(phase))
    omega = np.squeeze(np.asarray(omega))

    resp = mag * np.exp(1j * phase)
    nyq = np.real(resp) + 1j * np.abs(np.imag(resp))

    mask = np.isfinite(np.real(nyq)) & np.isfinite(np.imag(nyq))
    return omega[mask], nyq[mask]




# =========================
# Plot styling helpers
# =========================

def _set_origin_axes(ax):
    """Approximate Julia's framestyle=:origin."""
    ax.spines["left"].set_position("zero")
    ax.spines["bottom"].set_position("zero")
    ax.spines["right"].set_color("none")
    ax.spines["top"].set_color("none")
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")
    # ax.set_aspect("equal", adjustable="box")


def plot_circle(ax=None, n=400, **kwargs):
    """Plot unit circle in the disk."""
    if ax is None:
        ax = plt.gca()
    t = np.linspace(0, 2 * np.pi, n)
    ax.plot(np.cos(t), np.sin(t), **kwargs)


# =========================
# SRG construction
# =========================

def _compute_srg_data(sys, omega=None, n=2000, arc_points=120):
    if getattr(sys, "ninputs", 1) != 1 or getattr(sys, "noutputs", 1) != 1:
        raise ValueError("Only SISO systems are supported.")

    if omega is None:
        omega = _default_omega(sys, n=n)
    else:
        omega = np.asarray(omega, dtype=float)

    omega, nyq = _upper_nyquist(sys, omega)
    nyq_p = beltrami_klein(nyq)

    hull = _convex_hull_complex(nyq_p)
    if hull.size == 0:
        raise ValueError("No valid Nyquist points were produced.")

    hull_closed = np.r_[hull, hull[0]]
    srg_vertices = beltrami_klein_inverse(hull_closed)
    srg_upper_dense = _densify_closed_arc_chain(srg_vertices, arc_points=arc_points)

    return {
        "omega": omega,
        "nyq_upper": nyq,
        "nyq_lower": np.conj(nyq),
        "nyq_disk": nyq_p,
        "srg_vertices_upper": srg_vertices,
        "srg_upper_dense": srg_upper_dense,
        "srg_lower_dense": np.conj(srg_upper_dense),
        "srg_fill_polygon": _make_symmetric_polygon(srg_upper_dense),
        "srg_disk_boundary": hull_closed,
    }


# =========================
# Public plotting functions
# =========================

def _g_infinity(sys):
    """
    Limit of G(s) as |s| -> infinity for a SISO transfer function.
    Strictly proper -> 0
    Proper (same degree) -> ratio of leading coefficients
    Improper -> no finite limit
    """
    num = np.asarray(sys.num[0][0], dtype=float).flatten()
    den = np.asarray(sys.den[0][0], dtype=float).flatten()

    num = np.trim_zeros(num, trim="f")
    den = np.trim_zeros(den, trim="f")

    if len(num) < len(den):
        return 0.0 + 0.0j
    elif len(num) == len(den):
        return complex(num[0] / den[0])
    else:
        raise ValueError(
            "Improper transfer function: Nyquist infinity arc does not collapse to a finite point."
        )


def closed_nyquist_curve(sys, omega=None, omega_num=2000, **nyq_kwargs):
    """
    Build a visually closed Nyquist curve.

    For real-coefficient systems, the negative-frequency branch is the
    conjugate mirror of the positive-frequency branch.
    """
    if omega is None:
        omega = _default_omega(sys, n=omega_num)

    # positive-frequency response
    mag, phase, omega = ct.frequency_response(sys, omega=omega)
    mag = np.squeeze(np.asarray(mag))
    phase = np.squeeze(np.asarray(phase))
    primary = mag * np.exp(1j * phase)

    mirror = np.conj(primary[::-1])
    g_inf = _g_infinity(sys)

    # visually close the curve through the high-frequency limit
    full = np.r_[primary, g_inf, mirror]
    return primary, mirror, full


def srg_lti(
    sys,
    omega=None,
    n=5000,
    arc_points=120,
    ax=None,
    show_lower=True,
    shade=True,
    shade_alpha=0.20,
    plot_nyquist=True,
    nyquist_as_line=True,
    show_closed_nyquist=True,
    srg_color="cornflowerblue",
    nyq_color="tab:red",
    closed_nyq_color="crimson",
):
    """
    Plot SRG in the complex plane.

    Parameters
    ----------
    show_closed_nyquist : bool
        If True, overlay the visually closed Nyquist loop
        (+jw branch -> G(inf) -> -jw branch).
        This is for display only; it is NOT used to construct the SRG.
    """
    data = _compute_srg_data(sys, omega=omega, n=n, arc_points=arc_points)

    if ax is None:
        _, ax = plt.subplots()

    # Filled SRG region
    if shade:
        poly = data["srg_fill_polygon"]
        ax.fill(
            np.real(poly),
            np.imag(poly),
            color=srg_color,
            alpha=shade_alpha,
            linewidth=0,
            zorder=1,
        )

    # Upper SRG boundary
    ax.plot(
        np.real(data["srg_upper_dense"]),
        np.imag(data["srg_upper_dense"]),
        color=srg_color,
        lw=1.8,
        zorder=1,
    )

    # Lower SRG boundary
    if show_lower:
        ax.plot(
            np.real(data["srg_lower_dense"]),
            np.imag(data["srg_lower_dense"]),
            color=srg_color,
            lw=1.8,
            zorder=1,
        )

    # Standard Nyquist overlay from sampled positive-frequency response
    if plot_nyquist and not show_closed_nyquist:
        if nyquist_as_line:
            ax.plot(
                np.real(data["nyq_upper"]),
                np.imag(data["nyq_upper"]),
                color=nyq_color,
                lw=1.2,
                zorder=2,
            )
            if show_lower:
                ax.plot(
                    np.real(data["nyq_lower"]),
                    np.imag(data["nyq_lower"]),
                    color=nyq_color,
                    lw=1.2,
                    zorder=2,
                )
        else:
            ax.plot(
                np.real(data["nyq_upper"]),
                np.imag(data["nyq_upper"]),
                ".",
                color=nyq_color,
                ms=3,
                zorder=2,
            )
            if show_lower:
                ax.plot(
                    np.real(data["nyq_lower"]),
                    np.imag(data["nyq_lower"]),
                    ".",
                    color=nyq_color,
                    ms=3,
                    zorder=2,
                )

    # Closed Nyquist overlay for visualization
    if show_closed_nyquist:
        primary, mirror, full_nyq = closed_nyquist_curve(sys, omega=omega, omega_num=n)

        if nyquist_as_line:
            ax.plot(
                np.real(full_nyq),
                np.imag(full_nyq),
                color=closed_nyq_color,
                lw=1.2,
                zorder=2,
            )
        else:
            ax.plot(
                np.real(primary),
                np.imag(primary),
                ".",
                color=closed_nyq_color,
                ms=3,
                zorder=2,
            )
            ax.plot(
                np.real(mirror),
                np.imag(mirror),
                ".",
                color=closed_nyq_color,
                ms=3,
                zorder=2,
            )
            ax.plot(
                [np.real(_g_infinity(sys))],
                [np.imag(_g_infinity(sys))],
                ".",
                color=closed_nyq_color,
                ms=5,
                zorder=2,
            )

    _set_origin_axes(ax)
    ax.set_xlabel("Re", loc="right")
    ax.set_ylabel("Im", loc="top")

    data["ax"] = ax
    return data

def srg_lti_poincare(
    sys,
    omega=None,
    n=2000,
    ax=None,
    shade=True,
    shade_alpha=0.20,
    plot_nyquist=True,
    srg_color="cornflowerblue",
    nyq_color="red",
):
    """
    Plot SRG in the Beltrami-Klein disk / Poincare-style view.

    Added beyond the earlier version:
    - filled hull region
    - full unit circle
    """
    data = _compute_srg_data(sys, omega=omega, n=n)

    if ax is None:
        _, ax = plt.subplots()

    if shade:
        bd = data["srg_disk_boundary"]
        ax.fill(
            np.real(bd),
            np.imag(bd),
            color=srg_color,
            alpha=shade_alpha,
            linewidth=0,
            zorder=1,
        )

    ax.plot(
        np.real(data["srg_disk_boundary"]),
        np.imag(data["srg_disk_boundary"]),
        color=srg_color,
        lw=1.8,
        zorder=3,
    )

    if plot_nyquist:
        ax.plot(
            np.real(data["nyq_disk"]),
            np.imag(data["nyq_disk"]),
            ".",
            color=nyq_color,
            ms=3,
            zorder=2,
        )

    plot_circle(ax=ax, color=srg_color, lw=1.2)
    _set_origin_axes(ax)
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel("Re")
    ax.set_ylabel("Im")

    data["ax"] = ax
    return data


# =========================
# Example
# =========================

if __name__ == "__main__":
    # k_d = 1.0
    # A_v = -k_d
    # B_v = 1
    # C_v = 1
    # D_v = 0
    # SS_v = ct.ss(A_v, B_v, C_v, D_v)

    # A_x = np.array([[0, 1], [0, -k_d]])
    # B_x = np.array([[0], [1]])
    # C_x = np.array([[1, 0]])
    # D_x = np.array([[0]])
    # SS_x = ct.ss(A_x, B_x, C_x, D_x)

    # G_x = ct.tf(SS_x)

    # fig, ax = plt.subplots(figsize=(6, 6))
    # srg_lti(G_x, shade=True, show_lower=True, ax=ax)
    # ax.grid(which='minor', linestyle=':', linewidth='0.5', color='gray', alpha=0.5)

    # plt.figure()
    # ct.nyquist_plot(G_x, omega_num=5000)
    # plt.grid(which='minor', linestyle=':', linewidth='0.5', color='gray', alpha=0.5)
    # # plt.figure()
    # # srg_lti_poincare(G, shade=True)

    # G_v = ct.tf(SS_v)

    # fig, ax = plt.subplots(figsize=(6, 6))
    # srg_lti(G_v, shade=True, show_lower=True, ax=ax)
    # ax.grid(which='minor', linestyle=':', linewidth='0.5', color='gray', alpha=0.5)

    # plt.figure()
    # ct.nyquist_plot(G_v)
    # plt.grid(which='minor', linestyle=':', linewidth='0.5', color='gray', alpha=0.5)
    
    k = 1
    b = 1
    m = 1

    A_x = np.array([[0, 1], [-k/m, -b/m]])
    B_x = np.array([[0], [1/m]])
    C_x = np.eye(2)
    D_x = np.array([[0], [0]])
    SS_x = ct.ss(A_x, B_x, C_x, D_x)

    Q = np.eye(2)
    R = np.eye(1)
    K, S, E = ct.lqr(SS_x, Q, R)

    B_x_cl = B_x @ K
    D_x_cl = 0 * np.eye(2)

    SS_x_cl = ct.ss(A_x, B_x_cl, C_x, D_x_cl)

    G_x = ct.tf(SS_x)
    L_x = ct.tf(SS_x_cl)

    D = L_x[0, 0] * L_x[1, 1] - L_x[0, 1] * L_x[1, 0] # type: ignore

    print(D)


    fig, ax = plt.subplots(figsize=(6, 6))
    srg_lti(D, shade=True, show_lower=True, ax=ax)
    ax.grid(which='minor', linestyle=':', linewidth='0.5', color='gray', alpha=0.5)

    # plt.figure()
    # ct.nyquist_plot(G_x, omega_num=5000)
    # plt.grid(which='minor', linestyle=':', linewidth='0.5', color='gray', alpha=0.5)
    # plt.figure()
    # srg_lti_poincare(G, shade=True)

    plt.show()