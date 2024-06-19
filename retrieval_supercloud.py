import astropy.io as io
import astropy.units as u
import itertools
import json
import logging
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import speclib as sl
import sys
import ultranest
import ultranest.stepsampler
from astropy.table import Table
from astropy.time import Time
from astropy.convolution import Gaussian1DKernel, convolve
from specutils import Spectrum1D
from scipy import stats


def transform_uniform(quantile, a, b):

    return a + (b - a) * quantile


def transform_loguniform(quantile, a, b):
    la = np.log(a)
    lb = np.log(b)

    return np.exp(la + quantile * (lb - la))


def get_normal_prior(mu, sigma):

    return stats.norm(loc=mu, scale=sigma)


def get_truncated_normal_prior(mu, sigma, a, b):
    a_sigma, b_sigma = (a - mu) / sigma, (b - mu) / sigma

    return stats.truncnorm(loc=mu, scale=sigma, a=a_sigma, b=b_sigma)


def transform_dirichlet(quantiles):
    # https://en.wikipedia.org/wiki/Dirichlet_distribution#Random_variate_generation
    # https://johannesbuchner.github.io/UltraNest/priors.html
    gamma_quantiles = -np.log(quantiles)

    return gamma_quantiles / gamma_quantiles.sum()


def prior_transform1(quantiles):
    params = quantiles.copy()

    # Uniform priors for all temperatures
    params[0] = transform_uniform(quantiles[0], teff_bds[0], teff_bds[1])

    # Normal prior on rstar
    params[1] = rstar_prior.ppf(quantiles[1])

    # Normal prior on dist
    params[2] = dist_prior.ppf(quantiles[2])

    # Uniform prior on scale
    params[3] = transform_uniform(quantiles[3], scale_bds[0], scale_bds[1])

    # Uniform prior on logf
    params[4] = transform_uniform(quantiles[4], logf_bds[0], logf_bds[1])

    return params


def prior_transform2(quantiles):
    params = quantiles.copy()

    # Uniform priors for all temperatures
    params[0] = transform_uniform(quantiles[0], teff_bds[0], teff_bds[1])

    # Uniform priors for all temperatures
    params[1] = transform_uniform(quantiles[1], teff_bds[0], teff_bds[1])

    # f1, f2
    params[2 : 2 + 2] = transform_dirichlet(quantiles[2 : 2 + 2])

    # Normal prior on rstar
    params[4] = rstar_prior.ppf(quantiles[4])

    # Normal prior on dist
    params[5] = dist_prior.ppf(quantiles[5])

    # Uniform prior on scale
    params[6] = transform_uniform(quantiles[6], scale_bds[0], scale_bds[1])

    # Uniform prior on logf
    params[7] = transform_uniform(quantiles[7], logf_bds[0], logf_bds[1])

    return params


def prior_transform3(quantiles):
    params = quantiles.copy()

    # Uniform priors for all temperatures
    params[0] = transform_uniform(quantiles[0], teff_bds[0], teff_bds[1])

    # Uniform priors for all temperatures
    params[1] = transform_uniform(quantiles[1], teff_bds[0], teff_bds[1])

    # Uniform priors for all temperatures
    params[2] = transform_uniform(quantiles[2], teff_bds[0], teff_bds[1])

    # f1, f2, f3
    params[3 : 3 + 3] = transform_dirichlet(quantiles[3 : 3 + 3])

    # Normal prior on rstar
    params[6] = rstar_prior.ppf(quantiles[6])

    # Normal prior on dist
    params[7] = dist_prior.ppf(quantiles[7])

    # Uniform prior on scale
    params[8] = transform_uniform(quantiles[8], scale_bds[0], scale_bds[1])

    # Uniform prior on logf
    params[9] = transform_uniform(quantiles[9], logf_bds[0], logf_bds[1])

    return params


def prior_transform4(quantiles):
    params = quantiles.copy()

    # Uniform priors for all temperatures
    params[0] = transform_uniform(quantiles[0], teff_bds[0], teff_bds[1])

    # Uniform priors for all temperatures
    params[1] = transform_uniform(quantiles[1], teff_bds[0], teff_bds[1])

    # Uniform priors for all temperatures
    params[2] = transform_uniform(quantiles[2], teff_bds[0], teff_bds[1])

    # Uniform priors for all temperatures
    params[3] = transform_uniform(quantiles[3], teff_bds[0], teff_bds[1])

    # f1, f2, f3, f4
    params[4 : 4 + 4] = transform_dirichlet(quantiles[4 : 4 + 4])

    # Normal prior on rstar
    params[8] = rstar_prior.ppf(quantiles[8])

    # Normal prior on dist
    params[9] = dist_prior.ppf(quantiles[9])

    # Uniform prior on scale
    params[10] = transform_uniform(quantiles[10], scale_bds[0], scale_bds[1])

    # Uniform prior on logf
    params[11] = transform_uniform(quantiles[11], logf_bds[0], logf_bds[1])

    return params


def eval_model1(params):
    t1, rstar, dist, scale, _ = params
    flux1 = sg.get_spectrum(t1, logg, feh)

    # Flux at stellar surface
    y_model_star = flux1

    # Flux density at Earth
    y_model_earth = y_model_star * (rstar / (dist * pc_to_rsun)) ** 2

    # Set resolution
    spec = Spectrum1D(spectral_axis=xobs, flux=y_model_earth)
    kernel = Gaussian1DKernel(stddev=2)  # 2 pixels gives roughly R~100 at 3 µm
    flux_smooth = convolve(spec, kernel, boundary="extend") * u.Unit(
        "erg/(cm^2 * s * Å)"
    )
    scaled_flux_smooth = flux_smooth * scale

    return scaled_flux_smooth


def eval_model2(params):
    t1, t2, f1, f2, rstar, dist, scale, _ = params
    flux1 = sg.get_spectrum(t1, logg, feh)
    flux2 = sg.get_spectrum(t2, logg, feh)

    # Flux at stellar surface
    y_model_star = f1 * flux1 + f2 * flux2

    # Flux density at Earth
    y_model_earth = y_model_star * (rstar / (dist * pc_to_rsun)) ** 2

    # Set resolution
    spec = Spectrum1D(spectral_axis=xobs, flux=y_model_earth)
    kernel = Gaussian1DKernel(stddev=2)  # 2 pixels gives roughly R~100 at 3 µm
    flux_smooth = convolve(spec, kernel, boundary="extend") * u.Unit(
        "erg/(cm^2 * s * Å)"
    )
    scaled_flux_smooth = flux_smooth * scale

    return scaled_flux_smooth


def eval_model3(params):
    t1, t2, t3, f1, f2, f3, rstar, dist, scale, _ = params
    flux1 = sg.get_spectrum(t1, logg, feh)
    flux2 = sg.get_spectrum(t2, logg, feh)
    flux3 = sg.get_spectrum(t3, logg, feh)

    # Flux at stellar surface
    y_model_star = f1 * flux1 + f2 * flux2 + f3 * flux3

    # Flux density at Earth
    y_model_earth = y_model_star * (rstar / (dist * pc_to_rsun)) ** 2

    # Set resolution
    spec = Spectrum1D(spectral_axis=xobs, flux=y_model_earth)
    kernel = Gaussian1DKernel(stddev=2)  # 2 pixels gives roughly R~100 at 3 µm
    flux_smooth = convolve(spec, kernel, boundary="extend") * u.Unit(
        "erg/(cm^2 * s * Å)"
    )
    scaled_flux_smooth = flux_smooth * scale

    return scaled_flux_smooth


def eval_model4(params):
    t1, t2, t3, t4, f1, f2, f3, f4, rstar, dist, scale, _ = params
    flux1 = sg.get_spectrum(t1, logg, feh)
    flux2 = sg.get_spectrum(t2, logg, feh)
    flux3 = sg.get_spectrum(t3, logg, feh)
    flux4 = sg.get_spectrum(t4, logg, feh)

    # Flux at stellar surface
    y_model_star = f1 * flux1 + f2 * flux2 + f3 * flux3 + f4 * flux4

    # Flux density at Earth
    y_model_earth = y_model_star * (rstar / (dist * pc_to_rsun)) ** 2

    # Set resolution
    spec = Spectrum1D(spectral_axis=xobs, flux=y_model_earth)
    kernel = Gaussian1DKernel(stddev=2)  # 2 pixels gives roughly R~100 at 3 µm
    flux_smooth = convolve(spec, kernel, boundary="extend") * u.Unit(
        "erg/(cm^2 * s * Å)"
    )
    scaled_flux_smooth = flux_smooth * scale

    return scaled_flux_smooth


def loglikelihood1(params):
    # Variance is underestimated by constant amount
    # https://emcee.readthedocs.io/en/stable/tutorials/line/
    y_model = eval_model1(params)
    sigma2 = yerr.value**2 + y_model.value**2 * np.exp(2 * params[-1])
    like = -0.5 * np.sum(
        (yobs.value - y_model.value) ** 2 / sigma2 + np.log(2 * np.pi * sigma2)
    )

    return like


def loglikelihood2(params):
    # Variance is underestimated by constant amount
    # https://emcee.readthedocs.io/en/stable/tutorials/line/
    f1, f2 = params[2 : 2 + 2]
    if not f1 > f2:
        # Slope encourages minimzation of f2
        return -1e100 * (1 + f2)

    y_model = eval_model2(params)
    sigma2 = yerr.value**2 + y_model.value**2 * np.exp(2 * params[-1])
    like = -0.5 * np.sum(
        (yobs.value - y_model.value) ** 2 / sigma2 + np.log(2 * np.pi * sigma2)
    )

    return like


def loglikelihood3(params):
    # Variance is underestimated by constant amount
    # https://emcee.readthedocs.io/en/stable/tutorials/line/
    f1, f2, f3 = params[3 : 3 + 3]
    if not (f1 > f2 > f3):
        # Slope encourages minimzation of f2, f3
        return -1e100 * (1 + f2 + f3)

    y_model = eval_model3(params)
    sigma2 = yerr.value**2 + y_model.value**2 * np.exp(2 * params[-1])
    like = -0.5 * np.sum(
        (yobs.value - y_model.value) ** 2 / sigma2 + np.log(2 * np.pi * sigma2)
    )

    return like


def loglikelihood4(params):
    # Variance is underestimated by constant amount
    # https://emcee.readthedocs.io/en/stable/tutorials/line/
    f1, f2, f3, f4 = params[4 : 4 + 4]
    if not (f1 > f2 > f3 > f4):
        # Slope encourages minimzation of f2, f3, f4
        return -1e100 * (1 + f2 + f3 + f4)

    y_model = eval_model4(params)
    sigma2 = yerr.value**2 + y_model.value**2 * np.exp(2 * params[-1])
    like = -0.5 * np.sum(
        (yobs.value - y_model.value) ** 2 / sigma2 + np.log(2 * np.pi * sigma2)
    )

    return like


def calc_epsilon_from_posterior_sample1(row):
    flux1 = sg.get_spectrum(row["t1"], logg, feh)

    in_chord = flux1
    full_disk = flux1
    epsilon = in_chord / full_disk

    return epsilon.value


def calc_epsilon_from_posterior_sample2(row):
    flux1 = sg.get_spectrum(row["t1"], logg, feh)
    flux2 = sg.get_spectrum(row["t2"], logg, feh)

    f1 = row["f1"]
    f2 = row["f2"]

    in_chord = flux1
    full_disk = f1 * flux1 + f2 * flux2
    epsilon = in_chord / full_disk

    return epsilon.value


def calc_epsilon_from_posterior_sample3(row):
    flux1 = sg.get_spectrum(row["t1"], logg, feh)
    flux2 = sg.get_spectrum(row["t2"], logg, feh)
    flux3 = sg.get_spectrum(row["t3"], logg, feh)

    f1 = row["f1"]
    f2 = row["f2"]
    f3 = row["f3"]

    in_chord = flux1
    full_disk = f1 * flux1 + f2 * flux2 + f3 * flux3
    epsilon = in_chord / full_disk

    return epsilon.value


def calc_epsilon_from_posterior_sample4(row):
    flux1 = sg.get_spectrum(row["t1"], logg, feh)
    flux2 = sg.get_spectrum(row["t2"], logg, feh)
    flux3 = sg.get_spectrum(row["t3"], logg, feh)
    flux4 = sg.get_spectrum(row["t4"], logg, feh)

    f1 = row["f1"]
    f2 = row["f2"]
    f3 = row["f3"]
    f4 = row["f4"]

    in_chord = flux1
    full_disk = f1 * flux1 + f2 * flux2 + f3 * flux3 + f4 * flux4
    epsilon = in_chord / full_disk

    return epsilon.value



FileName = sys.argv[1]
NCOMPs = int(sys.argv[2])
FileName2Compare = FileName.split("/")[-1].replace("-","").replace("_","")[:-4].upper()


# Get the current time in ISO format using astropy
start_time = Time.now()
start_time_iso = start_time.isot.replace(":", "_")

# Configure logging
logging.basicConfig(
    filename=f"{start_time_iso}_{FileName2Compare}_{NCOMPs}.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)



#Fitting is done with 1, 2, 3 or 4 components.
assert NCOMPs==1 or NCOMPs==2 or NCOMPs==3 or NCOMPs==4
#Get temperature, MStar, RStar, LogG, FeH, Distance
Database = pd.read_csv("database/StellarParams.csv")


FileName2Compare = FileName.split("/")[-1].replace("-","").replace("_","")[:-4].upper()


Index = np.nan

for Counter,Target in enumerate(Database['Target']):
    if Target.upper()==FileName2Compare:
        Index = Counter
        break
    elif Target.upper()==FileName2Compare[:-1]: #If there is A or B in the filename
        Index = Counter
        break


if np.isnan(Index):
    print("The stellar parameter database mismatch, there should not be any A or B in the ")
    exit()
else:
    Temp = float(Database['Temp'].iloc[Index])
    TempErr = float(Database['TempErr'].iloc[Index])
    MStar = float(Database['Mass'].iloc[Index])
    MStarErr = float(Database['MassErr'].iloc[Index])
    RStar = float(Database['Radius'].iloc[Index])
    RStarErr = float(Database['RadiusErr'].iloc[Index])
    LogG = float(Database['LogG'].iloc[Index])
    LogGErr = float(Database['LogGErr'].iloc[Index])
    FeH = float(Database['FeH'].iloc[Index])
    FeHErr = float(Database['FeHErr'].iloc[Index])
    Distance = float(Database['Dist'].iloc[Index])
    DistanceErr = float(Database['DistErr'].iloc[Index])


print("Now running:", FileName)
# Main procedure
try:
    # Log the start time
    logging.info(f"Script started")

    # Log some package versions
    #logging.info(f"speclib version {sl.__version__}")
    #logging.info(f"ultranest version {ultranest.__version__}")

    # Define the fit parameters
    model_grid = "phoenix"
    ncomp = NCOMPs  # 1, 2, 3, or 4
    run_num = 1

    # Stellar parameters (Bouchy et al. 2005)
    lit_teff = (Temp, TempErr)
    lit_feh = (FeH, FeHErr)
    lit_mstar = (MStar, MStarErr) * u.M_sun
    lit_rstar = (RStar, RStarErr) * u.R_sun
    lit_logg = (LogG, LogGErr)
    # Gaia DR3
    lit_dist = (Distance, DistanceErr) * u.parsec

    # Boundaries for uniform priors
    rstar_bds = (0.08, 2.00)
    scale_bds = (0.1, 10.0)
    logf_bds = (-50, 0)
    
    # Other priors
    feh = 0.0  # fixed
    logg = lit_logg[0]
    rstar_prior = get_normal_prior(*lit_rstar)  # Normal(mu, sigma) (R_sun)
    dist_prior = get_normal_prior(*lit_dist)  # Normal(mu, sigma) (pc)

    # Speed up the calculation by defining this beforehand
    pc_to_rsun = (u.pc / u.R_sun).decompose().scale
    
    print("Reached at line 466")
    teff_bds_dict = {
        "phoenix": (2300, 10000),
        "mps-atlas": (3500, 6000),
        "sphinx": (2000, 4000),
    }
    param_names_dict = {
        1: ["t1", "rstar", "dist", "scale", "logf"],
        2: ["t1", "t2", "f1", "f2", "rstar", "dist", "scale", "logf"],
        3: ["t1", "t2", "t3", "f1", "f2", "f3", "rstar", "dist", "scale", "logf"],
        4: [
            "t1",
            "t2",
            "t3",
            "t4",
            "f1",
            "f2",
            "f3",
            "f4",
            "rstar",
            "dist",
            "scale",
            "logf",
        ],
    }
    prior_transform_dict = {
        1: prior_transform1,
        2: prior_transform2,
        3: prior_transform3,
        4: prior_transform4,
    }
    eval_model_dict = {
        1: eval_model1,
        2: eval_model2,
        3: eval_model3,
        4: eval_model4,
    }
    loglikelihood_dict = {
        1: loglikelihood1,
        2: loglikelihood2,
        3: loglikelihood3,
        4: loglikelihood4,
    }
    calc_epsilon_dict = {
        1: calc_epsilon_from_posterior_sample1,
        2: calc_epsilon_from_posterior_sample2,
        3: calc_epsilon_from_posterior_sample3,
        4: calc_epsilon_from_posterior_sample4,
    }

    # Wavelength array and data
    spec_file = FileName
    d = io.ascii.read(FileName)
    xobs = d["wave"] * u.AA
    yobs = d["flux"] * u.Unit("erg/(cm^2*s)")*1./u.AA
    yerr = d["flux_err"] * u.Unit("erg/(cm^2*s)")*1./u.AA
    # # Suppressing outputs
    # logger = logging.getLogger("ultranest")
    # handler = logging.StreamHandler(sys.stdout)
    # handler.setLevel(logging.WARNING)
    # formatter = logging.Formatter("[ultranest] [%(levelname)s] %(message)s")
    # handler.setFormatter(formatter)
    # logger.addHandler(handler)
    # logger.setLevel(logging.WARNING)
    
    # Load models
    teff_bds = teff_bds_dict[model_grid]
    logging.info(f"Loading {model_grid} spectral grid...")
    print("Now loading the cross-sections:")
    sg = sl.SpectralGrid(teff_bds, logg, feh, xobs, model_grid=model_grid)
    print("Loading cross-sections finished:")   
    # Make output directory
    out_dir = f"results_{FileName2Compare}/{model_grid}/ncomp{ncomp}/"
    if not os.path.exists(out_dir):
        logging.info(f"Making output directory {out_dir}")
        os.makedirs(out_dir)

    # Skip completed runs
    sampling = True
    csv_file = out_dir + f"run{run_num}/info/post_summary.csv"
    if os.path.exists(csv_file):
        logging.info(f"Found {csv_file}. Skipping retrieval...")
        sampling = False

    # Define model
    param_names = param_names_dict[ncomp]
    ndim = len(param_names)
    ndata = len(xobs)
    prior_transform = prior_transform_dict[ncomp]
    eval_model = eval_model_dict[ncomp]
    loglikelihood = loglikelihood_dict[ncomp]

    if sampling:
        # Perform the sampling
        logging.info(f"Sampling")
        sampler = ultranest.ReactiveNestedSampler(
            param_names,
            loglikelihood,
            prior_transform,
            log_dir=out_dir,
            run_num=run_num,
        )
        nsteps = 10 * ndim
        sampler.stepsampler = ultranest.stepsampler.SliceSampler(
            nsteps=nsteps,
            generate_direction=ultranest.stepsampler.generate_mixture_random_direction,  # noqa
        )
        results = sampler.run(
            show_status=False,
            Lepsilon=0.01,
            # frac_remain=0.5,
            # max_num_improvement_loops=3,
        )

        # Save outputs
        try:
            sampler.plot_run()
            sampler.plot_trace()
            sampler.plot_corner()
        except ValueError:
            pass
    elif not sampling:
        results_file = out_dir + f"run{run_num}/info/results.json"
        with open(results_file, "r") as buffer:
            results = json.load(buffer)

    post = pd.read_csv(
        out_dir + f"run{run_num}/chains/equal_weighted_post.txt", delim_whitespace=True
    )

    logging.info(f"Summarizing results")
    # Fit to stellar spectrum
    models = np.array([eval_model(row) for i, row in post.iterrows()])
    mean_model = np.mean(models, axis=0)
    std_model = np.std(models, axis=0)
    x2s = np.array([((yobs.value - m) ** 2 / yerr.value**2).sum() for m in models])
    np.save(out_dir + f"run{run_num}/results/mean_model.npy", mean_model)
    np.save(out_dir + f"run{run_num}/results/std_model.npy", std_model)
    np.save(out_dir + f"run{run_num}/results/x2s.npy", x2s)
    np.random.seed(46 + 2)
    idx_subset = np.random.choice(models.shape[0], 1000, replace=False)
    np.save(
        out_dir + f"run{run_num}/results/posterior_models.npy", models[idx_subset, :]
    )

    # Residuals
    resid = yobs.value - mean_model
    resid_ppm = (resid / mean_model) * 1e6
    # Account for extra variance inferred via fit
    post_sigma_jit = np.exp(post["logf"].values)[:, np.newaxis] * models
    post_yerr = np.sqrt((yerr.value) ** 2 + post_sigma_jit**2).mean(axis=0)
    yerr_ppm = (post_yerr / mean_model) * 1e6

    # Goodness of fit
    gof = {x: results[x] for x in results if x in ("logz", "logzerr")}
    gof["logl"] = results["maximum_likelihood"]["logl"]
    gof["n"] = ndata
    gof["k"] = ndim
    gof["bic"] = gof["k"] * np.log(gof["n"]) - 2 * gof["logl"]
    gof["aic"] = 2 * gof["k"] - 2 * gof["logl"]
    gof["aicc"] = gof["aic"] + (2 * gof["k"] ** 2 + 2 * gof["k"]) / (
        gof["n"] - gof["k"] - 1
    )
    gof["x2"] = np.mean(x2s)  # ((yobs.value - mean_model)**2 / yerr.value**2).sum()
    gof["x2_std"] = np.std(x2s)
    gof["x2_infl"] = ((yobs.value - mean_model) ** 2 / post_yerr**2).sum()
    gof["dof"] = len(yobs) - ndim
    gof["x2r"] = gof["x2"] / gof["dof"]
    gof["x2r_std"] = gof["x2_std"] / gof["dof"]
    gof["x2r_infl"] = gof["x2_infl"] / gof["dof"]
    file_path = out_dir + f"run{run_num}/info/goodness_of_fit.json"
    with open(file_path, "w") as buffer:
        json.dump(gof, buffer)

    fig, ax = plt.subplots(constrained_layout=True)
    ax.fill_between(
        xobs.value,
        mean_model + std_model,
        mean_model - std_model,
        color="C0",
        alpha=0.5,
    )
    ax.errorbar(
        xobs.value, yobs.value, post_yerr, ls="", marker=".", color="k", mfc="white"
    )
    ax.set_xlabel("Wavelength (Å)")
    ax.set_ylabel("Flux at Earth (erg/s/cm$^2$/Å)")
    title = (
        out_dir + " | $x^2_r$ = "
        f"{gof['x2r']:.2f}" + " | $\log Z$ = "
        f"{gof['logz']:.2f} ± {gof['logzerr']:.2f}"
    )
    ax.set_title(title)
    plt.savefig(out_dir + f"run{run_num}/plots/model.pdf", bbox_inches="tight")

    # X2
    fig, ax = plt.subplots(constrained_layout=True)
    ax.hist(x2s, bins=100)
    ax.axvline(gof["x2"], color="k", ls="--", label="mean")
    ax.axvline(gof["x2"] + gof["x2_std"], color="k", ls=":", label="$\pm 1 \sigma$")
    ax.axvline(gof["x2"] - gof["x2_std"], color="k", ls=":")
    ax.legend()
    ax.set_xlabel("$\chi^2$")
    ax.set_ylabel("Samples")
    ax.set_title(out_dir)
    plt.savefig(out_dir + f"run{run_num}/plots/X2.pdf", bbox_inches="tight")

    # Epsilon
    calc_epsilon_from_posterior_sample = calc_epsilon_dict[ncomp]
    epsilon_models = np.array(
        [calc_epsilon_from_posterior_sample(row) for i, row in post.iterrows()]
    )
    mean_epsilon = np.mean(epsilon_models, axis=0)
    std_epsilon = np.std(epsilon_models, axis=0)
    np.save(out_dir + f"run{run_num}/extra/mean_epsilon.npy", mean_epsilon)
    np.save(out_dir + f"run{run_num}/extra/std_epsilon.npy", std_epsilon)
    # Save a subset
    np.random.seed(46 + 2)
    idx_subset = np.random.choice(epsilon_models.shape[0], 1000, replace=False)
    np.save(
        out_dir + f"run{run_num}/extra/posterior_epsilons.npy",
        epsilon_models[idx_subset, :],
    )

    fig, ax = plt.subplots()
    ax.fill_between(
        xobs.value,
        mean_epsilon + std_epsilon,
        mean_epsilon - std_epsilon,
        color="C0",
        alpha=0.5,
    )
    ax.plot(xobs, mean_epsilon, lw=0.2)
    ax.set_xlabel("Wavelength (Å)")
    ax.set_ylabel("Stellar contamination factor")
    title = (
        out_dir + " | $x^2_r$ = "
        f"{gof['x2r']:.2f}" + " | $\log Z$ = "
        f"{gof['logz']:.2f} ± {gof['logzerr']:.2f}"
    )
    ax.set_title(title)
    plt.savefig(out_dir + f"run{run_num}/plots/epsilon.pdf", bbox_inches="tight")

    plt.close("all")

except Exception as e:
    # Log exceptions
    logging.error(f"Error occurred: {e}")

finally:
    # Get the stop time
    stop_time = Time.now()

    # Calculate the execution duration
    execution_duration = (stop_time - start_time).sec

    # Log the stop time and execution duration
    logging.info(f"Script ended")
    logging.info(f"Script executed in {execution_duration:.2f} seconds")
