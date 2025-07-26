import adaptkde as ak
import pandas as pd
import numpy as np
from multiprocessing import Pool
import pickle
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import sys


def streamsHeating(t, sigma0, h, alpha):
    """
    t is in Myr
    sigma0 is in km/s
    h is dimensionless
    alpha is dimensionless
    """
    return (sigma0 ^ (1 / alpha) + h * t / 14000 * (35) ^ (1 / alpha)) ^ alpha


def compute(a):
    return ak.cvAdaptiveKDE(
        a, nfolds=10, kappa=0.0372, nu=0.0755, selector=np.mean
    ).kde


def est_veldisp(kde, rot_matrix, x, y, z):
    roto = Rotation.from_matrix(rot_matrix)

    vs = kde.draw_conditional_on_x((x, y, z), size=3000)
    v_rot = roto.apply(vs)

    return np.cov(v_rot.T)


def check_draw():
    df = pd.read_csv("df_iso_orbits.csv")
    header = ["t", "x_pos", "y_pos", "z_pos", "x_vel", "y_vel", "z_vel"]
    out = ak.standard_unit_transform_svector(
        df[header].sort_values("t").to_numpy()[:, 1:].T
    ).T
    xvvvs = np.split(out, 121, axis=0)
    # ax = plt.axes()
    # ax.scatter(xvvvs[0][:,0],xvvvs[0][:,1])
    # plt.savefig("spread.png")
    # plt.close()
    pt = xvvvs[0][1983, :3]
    kdes = list(map(compute, iter(xvvvs[:1])))
    # with Pool() as p:
    #     kdes = p.map(compute,iter(xvvvs[:1]))
    # kdes = p.map(ak.computeKDE,iter(xvvvs[:2]))
    print(kdes[0].scales, kdes[0].covalpha_overall)
    print(len(kdes[0].covariances), len(np.unique(kdes[0].covariances, axis=0)))
    N = 10000000
    ax = plt.axes()
    pts = kdes[0].draw(size=N)[:, :3]
    print(pts[:10])
    lengths = np.linalg.norm(pts - pt, axis=1)
    ax.scatter(
        pts[lengths < 10, 0],
        pts[lengths < 10, 1],
        label=f"proportion of draws{len(pts[lengths<10,0])/N}",
        s=1,
    )
    ax.legend()
    # ax.hist(lengths,40)
    plt.savefig("draw.png", dpi=600)


def ellipsoid(cvkde):
    N = 1000
    draws = cvkde.draw_marginal(size=N)
    eval_list = []
    evec_list = []
    for j in range(N):
        cov = est_veldisp(
            cvkde, np.eye(3), draws[j][0], draws[j][1], draws[j][2]
        )
        evals, evecs = np.linalg.eig(cov)
        eval_list.append(evals)
        evec_list.append(evecs)
    eval_ind = np.argsort(np.array(eval_list), axis=1)
    eval_list = np.take_along_axis(eval_list, eval_ind)
    eval_list = (eval_list.T / eval_list.T.min(axis=0)).T
    res = np.mean(eval_list, axis=0)[1:]
    return res


def sigma(cvkde):
    N = 1000
    draws = cvkde.draw_marginal(size=N)
    eval_list = []
    for j in range(N):
        cov = est_veldisp(
            cvkde, np.eye(3), draws[j][0], draws[j][1], draws[j][2]
        )
        evals, _ = np.linalg.eig(cov)
        eval_list.append(np.mean(evals))
    return np.sqrt(np.mean(eval_list))


TIMESTEPS = 121


def main():
    df = pd.read_csv("df_iso_orbits.csv")
    header = ["t", "x_pos", "y_pos", "z_pos", "x_vel", "y_vel", "z_vel"]
    out = ak.standard_unit_transform_svector(
        df[header].sort_values("t").to_numpy()[:, 1:].T
    ).T
    xvvvs = np.split(out, 121, axis=0)
    kdes = list(map(compute, iter(xvvvs[:TIMESTEPS])))
    print(kdes)
    # kdes = [ak.computeKDE(frame) for frame in xvvvs]
    # with Pool() as p:
    # kdes = p.map(compute,iter(xvvvs[:TIMESTEPS]))
    # kdes = p.map(ak.computeKDE,iter(xvvvs[:TIMESTEPS]))
    with open("kde.pickle", "wb") as file:
        pickle.dump(kdes, file)


def compute_sigma():
    with open("kde.pickle", "rb") as file:
        kdes = pickle.load(file)
    with Pool() as p:
        # sigmas = p.map(sigma, kdes)
        sigmas = p.map(ak.sigma, kdes)
    print(sigmas)
    with open("sigmas.pickle", "wb") as file:
        pickle.dump(sigmas, file)


def compute_ellipsoid():
    with open("kde.pickle", "rb") as file:
        kdes = pickle.load(file)
    with Pool() as p:
        ells = p.map(ellipsoid, kdes)
    print(ells)
    with open("ells.pickle", "wb") as file:
        pickle.dump(ells, file)


def plot_ells():
    with open("ells.pickle", "rb") as file:
        ells = pickle.load(file)
        ells = np.array(ells)
    ax = plt.axes()
    ax.plot(ells.T[0, :], ells.T[1, :])
    plt.savefig("ells.png")
    plt.close()
    ax = plt.axes()
    ax.plot(
        np.linspace(0, 25 * (TIMESTEPS - 1), TIMESTEPS),
        ells.T[1, :] / ells.T[0, :],
        color="k",
    )
    ax.plot(
        np.linspace(0, 25 * (TIMESTEPS - 1), TIMESTEPS), ells.T[1, :], color="r"
    )
    ax.plot(
        np.linspace(0, 25 * (TIMESTEPS - 1), TIMESTEPS), ells.T[0, :], color="b"
    )
    ax.set_yscale("log")
    plt.savefig("ells_ratio.png")


def plot_sigma():
    with open("sigmas.pickle", "rb") as file:
        sigmas = pickle.load(file)
    ax = plt.axes()
    ax.plot(np.linspace(0, 25 * (TIMESTEPS - 1), TIMESTEPS), sigmas)
    plt.savefig("sigmas_test.png")


def plot_sigmas():
    with open("sigmas.pickle", "rb") as file:
        sigmas = pickle.load(file)
    with open("sigmas1.pickle", "rb") as file:
        sigmas1 = pickle.load(file)
    with open("sigmas2.pickle", "rb") as file:
        sigmas2 = pickle.load(file)
    with open("sigmas4.pickle", "rb") as file:
        sigmas4 = pickle.load(file)
    with open("sigmas_unheated.pickle", "rb") as file:
        sigmas_unheated = pickle.load(file)
    ax = plt.axes()
    ax.plot(np.linspace(0, 25 * (TIMESTEPS - 1), TIMESTEPS), sigmas)
    ax.plot(np.linspace(0, 25 * (TIMESTEPS - 1), TIMESTEPS), sigmas1)
    ax.plot(np.linspace(0, 25 * (TIMESTEPS - 1), TIMESTEPS), sigmas2)
    ax.plot(
        np.linspace(0, 25 * (TIMESTEPS - 1), TIMESTEPS),
        sigmas_unheated,
        linestyle="--",
    )
    ax.plot(np.linspace(0, 25 * (TIMESTEPS - 1), TIMESTEPS), sigmas4)
    ax.set_ylabel("velocity dispersion")
    ax.set_xlabel("time in Myr")
    plt.savefig("sigmas_test.png")


def test_draw():
    with open("kde.pickle", "rb") as file:
        kdes = pickle.load(file)
    N = 100000
    fig, ax = plt.subplots(2, 2)
    ax[0, 0].hist(
        np.linalg.norm(kdes[1].draw_marginal(size=N)[:, :2], axis=1), 40
    )
    ax[1, 0].hist(kdes[1].draw_marginal(size=N)[:, 2], 40)
    ax[0, 1].hist(np.linalg.norm(kdes[1].draw(size=N)[:, 3:5], axis=1), 40)
    ax[1, 1].hist(kdes[1].draw(size=N)[:, 5], 40)
    plt.savefig("distribution.png")


def printout_hyperparam():
    with open("kde.pickle", "rb") as file:
        kdes = pickle.load(file)
    print([kde.covalpha_overall for kde in kdes])
    ax = plt.axes()
    ax.plot(
        np.linspace(0, 25 * (TIMESTEPS - 1), TIMESTEPS),
        [kde.covalpha_overall for kde in kdes],
    )
    plt.savefig("param.png")


if __name__ == "__main__":
    # main()
    # compute_sigma()
    # check_draw()
    # compute_ellipsoid()
    # plot_ells()
    # plot_sigma()
    plot_sigmas()
    # printout_hyperparam()
    # test_draw()
