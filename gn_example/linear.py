import numpy as np

N = np.ndarray
TangentN = np.ndarray
M = np.ndarray
TangentM = np.ndarray


def ominus_n(lhs: N, rhs: N) -> TangentN:
    return np.asarray(lhs - rhs)


def oplus_m(lhs: M, rhs: TangentM) -> M:
    return np.asarray(lhs + rhs)


# from https://en.wikipedia.org/wiki/Gauss%E2%80%93Newton_algorithm#Example
s = np.array([0.038, 0.194, 0.425, 0.626, 1.253, 2.500, 3.740])
z_bar = np.array([0.050, 0.127, 0.094, 0.2122, 0.2729, 0.2665, 0.3317])
prec = np.eye(len(z_bar))


def f(x: M) -> N:
    return np.asarray(x[0] * s / (x[1] + s))


def e(x: M) -> float:
    eps = ominus_n(z_bar, f(x))
    return float(0.5 * eps.T @ prec @ eps)


def jacobian(x: M) -> np.ndarray:
    ret = np.array(
        [
            -s / (x[1] + s),
            x[0] * s / (x[1] + s) ** 2,
        ]
    ).T
    n = len(s)
    m = len(x)
    assert ret.shape == (n, m)
    return ret


x = np.array([0.9, 0.2])
print(f"init, x={x}")
for k in range(5):
    eps = ominus_n(z_bar, f(x))
    j = jacobian(x)
    tau = np.linalg.lstsq(j.T @ prec @ j, -j.T @ prec @ eps, rcond=None)[0]
    x = oplus_m(x, tau)
    print(f"iter={k}, x={x}, tau={tau}, e={e(x)}")
