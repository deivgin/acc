import numpy as np


def ned_to_body(
    v_north: np.ndarray,
    v_east: np.ndarray,
    v_down: np.ndarray,
    phi: np.ndarray,
    theta: np.ndarray,
    psi: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rotate NED velocity to body frame using 3-2-1 Euler rotation.

    Returns (u, v, w) body-frame velocity components.
    """
    cp = np.cos(phi)
    sp = np.sin(phi)
    ct = np.cos(theta)
    st = np.sin(theta)
    cs = np.cos(psi)
    ss = np.sin(psi)

    # DCM rows (NED -> body)
    u = ct * cs * v_north + ct * ss * v_east - st * v_down
    v = (
        (sp * st * cs - cp * ss) * v_north
        + (sp * st * ss + cp * cs) * v_east
        + sp * ct * v_down
    )
    w = (
        (cp * st * cs + sp * ss) * v_north
        + (cp * st * ss - sp * cs) * v_east
        + cp * ct * v_down
    )
    return u, v, w


def body_to_wind_forces(
    fx: np.ndarray,
    fy: np.ndarray,
    fz: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Transform body-frame forces to wind-frame (lift, drag, sideforce).

    Body to wind rotation via alpha (pitch) and beta (yaw):
      D = -fx·cos(α)·cos(β) - fy·sin(β) - fz·sin(α)·cos(β)
      Y = -fx·cos(α)·sin(β) + fy·cos(β) - fz·sin(α)·sin(β)
      L = fx·sin(α) - fz·cos(α)
    """
    ca = np.cos(alpha)
    sa = np.sin(alpha)
    cb = np.cos(beta)
    sb = np.sin(beta)

    drag = -fx * ca * cb - fy * sb - fz * sa * cb
    side = -fx * ca * sb + fy * cb - fz * sa * sb
    lift = fx * sa - fz * ca

    return lift, drag, side
