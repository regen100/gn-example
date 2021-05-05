from typing import Type

import numpy as np
from scipy.spatial.transform import Rotation

N = np.ndarray
TangentN = np.ndarray
M = Type[Rotation]
TangentM = np.ndarray


def ominus_n(lhs: N, rhs: N) -> TangentN:
    return np.asarray(lhs - rhs).reshape(-1)


def oplus_m(lhs: Rotation, rhs: TangentM) -> Rotation:
    return lhs * Rotation.from_rotvec(rhs)


np.random.seed(0)
s = np.random.random((3, 3))
x_gt: M = Rotation.random()
z_bar: N = x_gt.apply(s)
prec = np.eye(s.size)


def f(x: M) -> N:
    return np.asarray(x.apply(s))


def e(x: M) -> float:
    eps = ominus_n(z_bar, f(x))
    return float(0.5 * eps.T @ prec @ eps)


def jacobian(x: M) -> np.ndarray:
    ret = []
    for i in s:
        ret.append(
            x.as_matrix()
            @ np.array([[0, -i[2], i[1]], [i[2], 0, -i[0]], [-i[1], i[0], 0]]),
        )
    return np.vstack(ret)


x = Rotation.identity()
print(f"init, Log(x)={x.as_rotvec()}")
for k in range(5):
    eps = ominus_n(z_bar, f(x))
    j = jacobian(x)
    tau = np.linalg.lstsq(j.T @ prec @ j, -j.T @ prec @ eps, rcond=None)[0]
    x = oplus_m(x, tau)
    print(f"iter={k}, Log(x)={x.as_rotvec()}, tau={tau}, e={e(x)}")
print(f"Log(x_gt)={x_gt.as_rotvec()}")
