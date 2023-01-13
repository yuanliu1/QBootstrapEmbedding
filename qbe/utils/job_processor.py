"""
File containing functions to post process results from jobs
"""
# Add paths
import os
from typing import Dict, List, Any, Union, Tuple

import numpy as np
import matplotlib.pyplot as plt
import pickle

def plot_trend(rmse_learners, nqueries_learners, nqueries_lower_bound=None,
               rmse_lower_bound=None, error_learners=None, plotting_options=None):
    """
    Helper function for plotting trends of various measurement procedures

    Inputs (fairly self-explanatory):
    :param rmse_learners:
    :param nqueries_learners:
    :param nqueries_lower_bound:
    :param rmse_lower_bound:
    :param error_learners:
    :param plotting_options:
    :return:
    """
    # Default plotting options
    default_plotting_options = {'label_learners': ['Uniform CS', 'LBCS', 'APS'],
                                'FLAG_save_plot': True, 'save_filename': 'results.png',
                                'figsize_plot': (12, 10), 'skip_learner': 0,
                                'FLAG_fit_slopes': [True, True],
                                'n_iters_end': None, 'slope_scales': 0.9*np.ones(len(rmse_learners)),
                                'slope_visual': [False, 1.],
                                'FLAG_legend_outside': True, 'FLAG_chemical_accuracy': False}

    if plotting_options is None:
        plotting_options = default_plotting_options
    else:
        for _key in default_plotting_options.keys():
            plotting_options.setdefault(_key, default_plotting_options[_key])

    # Plot the trends
    # color_lines = ['r-', 'b-', 'g-', 'm-', 'r--', 'b--', 'g--', 'm--']
    marker_color = ['tab:red', 'tab:blue', 'tab:green', 'tab:purple',
                    'tab:orange', 'tab:cyan', 'tab:olive', 'tab:brown', 'tab:pink']
    marker_style = ['o', 's', 'd', 'v', '<', '>', '^', 'p', 'x']

    plt.figure(figsize=plotting_options['figsize_plot'])
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.family'] = "sans-serif"
    plt.rcParams['font.sans-serif'] = "Helvetica"
    plt.rcParams['figure.titlesize'] = 22
    plt.rcParams['axes.labelsize'] = 22
    plt.rcParams['axes.titlesize'] = 22
    plt.rcParams['legend.fontsize'] = 18
    plt.rcParams['xtick.labelsize'] = 18
    plt.rcParams['ytick.labelsize'] = 18

    # If we want to use the marker/color information starting from a different counter
    skip_learner = plotting_options['skip_learner']
    slope_scales = plotting_options['slope_scales']

    # Loop over the list of rmse for different learners
    for ind in range(len(rmse_learners)):
        rmse_ind = rmse_learners[ind]
        nqueries_ind = nqueries_learners[ind]

        # Plotting styles, labels, etc.
        color_learner_plot = marker_color[ind + skip_learner]
        marker_style_plot = marker_style[ind + skip_learner]
        label_learner = plotting_options['label_learners'][ind]

        n_iters_end = plotting_options['n_iters_end']
        if n_iters_end is None:
            n_iters_end = np.amax([round(len(nqueries_ind) / 3), 4])

        # RMSE of Learners (Data) with RMSE Errorbars if available
        if error_learners is not None:
            error_ind = error_learners[ind]

            if error_ind.shape[0] == 2:
                plt.fill_between(nqueries_ind, rmse_ind-error_ind[0,:], rmse_ind+error_ind[1,:],
                                 color=color_learner_plot, alpha=0.2)
            else:
                plt.fill_between(nqueries_ind, rmse_ind - error_ind, rmse_ind + error_ind,
                                 color=color_learner_plot, alpha=0.2)

            # Errorbar might not be working
            # plt.errorbar(nqueries_ind, rmse_ind, yerr=error_ind,
            #              fmt=marker_style[ind], mec=marker_color[ind], uplims=True, lolims=True)

        plt.plot(nqueries_ind, rmse_ind, color=color_learner_plot, marker=marker_style_plot, label=label_learner)

        # RMSE of Learners (Fits) -- beginning and ending of trends
        if plotting_options['FLAG_fit_slopes'][0]:
            poly_init, cov_init = np.polyfit(np.log(nqueries_ind[0:6]), np.log(rmse_ind[0:6]), 1, cov=True)
            label_fit_init = r"slope = %0.2f $\pm$ %0.2f" % (poly_init[0], np.sqrt(np.diag(cov_init)[0]))

            plt.plot(nqueries_ind[0:6], np.exp(np.polyval(poly_init,
                                                          np.log(nqueries_ind[0:6]))),
                     color=color_learner_plot, linestyle='dashed', label=label_fit_init)

        if plotting_options['FLAG_fit_slopes'][1]:
            poly_end, cov_end = np.polyfit(np.log(nqueries_ind[-n_iters_end:]),
                                           np.log(rmse_ind[-n_iters_end:]), 1, cov=True)

            label_fit_end = r"slope = %0.2f $\pm$ %0.2f" % (poly_end[0], np.sqrt(np.diag(cov_end)[0]))

            plt.plot(nqueries_ind[-n_iters_end:],
                     slope_scales[ind]*np.exp(np.polyval(poly_end, np.log(nqueries_ind[-n_iters_end:]))),
                     color=color_learner_plot, linestyle='dashed', label=label_fit_end)

    if rmse_lower_bound is not None and nqueries_lower_bound is not None:
        plt.plot(nqueries_lower_bound, rmse_lower_bound, 'k-', label='Cramer-Rao Bound')
        poly_crb, cov_crb = np.polyfit(np.log(nqueries_lower_bound), np.log(rmse_lower_bound), 1, cov=True)
        print(poly_crb)

    if plotting_options['FLAG_chemical_accuracy']:
        # plot a line representing ~1mH
        chemical_accuracy = 1.6e-3
        _x = np.linspace(0, 1e6, 50)
        plt.plot(_x, chemical_accuracy*np.ones(len(_x)), '--k')

    if plotting_options['slope_visual'][0]:
        # plot a line showing slope -0.5
        slope = -0.5
        if np.log10(np.amax(nqueries_learners[0])) > 5:
            _x = 10 ** np.linspace(4, 6, 10)
        else:
            _x = 10 ** np.linspace(3, 4, 10)

        _a = 10**(np.log10(plotting_options['slope_visual'][1]) + 2)
        _y = _a*(_x**slope)
        plt.plot(_x, _y, '--k', label='slope = -0.5')

    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel("Number of samples")

    plt.ylabel("RMSE (in Hartree)")
    # Limits
    # plt.ylim([0.5e-2, 2.5])
    #
    # # Tick-marks
    # plt.xticks([2e3,5e3,1e4,2e4,5e4,1e5])
    # plt.gca().set_xticklabels([r'$2 \times 10^{3}$', r'$5 \times 10^{3}$', r'$10^{4}$', r'$2 \times 10^{4}$',
    #                            r'$5 \times 10^{4}$', r'$10^{5}$'])
    #
    # plt.yticks([2e-2, 5e-2, 1e-1, 2e-1, 5e-1, 1e0, 2e0])
    # plt.gca().set_yticklabels([r'$2 \times 10^{-2}$', r'$5 \times 10^{-2}$', r'$10^{-1}$',
    #                            r'$2 \times 10^{-1}$', r'$5 \times 10^{-1}$', r'$10^{0}$', r'$2 \times 10^{0}$'])

    plt.grid(True)
    if plotting_options['FLAG_legend_outside']:
        plt.legend(bbox_to_anchor=(1.05, 1.025))
    else:
        #plt.legend(loc="upper right")
        plt.legend(loc='best')

    if plotting_options['FLAG_save_plot']:
        plt.savefig(plotting_options['save_filename'], bbox_inches='tight', pad_inches=0, dpi=300)


class EmpiricalEstimator(object):
    """
    Class with methods to carry out empirical estimation from log-files and pickle-files over different runs
    """
    def __init__(self, SAVE_DIR, max_iters=10, FLAG_filter=False, FLAG_rmse_test=False, do_plot=False):
        """
        Inputs:
            SAVE_DIR:
            max_iters, cumulative_samples: Usual meaning
            FLAG_filter:
            FLAG_rmse_test: Did the experiment that we ran involve computation of "Testing" RMSE
        """
        self.SAVE_DIR = SAVE_DIR
        self.max_iters = max_iters

        self.FLAG_filter = FLAG_filter

        # Load results from the save directory
        list_run_files = [os.path.join(SAVE_DIR, f) for f in os.listdir(SAVE_DIR) if f.endswith(".txt")]

        # Define RMSE over different runs
        rmse_job = []
        norm_grad_job = []
        neig_calls_job = []

        n_runs = 0

        # n_iter, n_eig_solver_calls,
        # self.norm_gradients[-1], self.rmse_error[-1],
        # self.run_time[-1]

        iter_col = 0
        neig_calls_col = 1
        norm_grad_col = 2
        rmse_col = 3

        for run_file in list_run_files:
            contents_run = np.loadtxt(run_file)
            if contents_run.shape[0] >= max_iters:
                neig_calls_run_temp = contents_run[0:max_iters, neig_calls_col]
                neig_calls_job.append(neig_calls_run_temp)

                rmse_run_temp = contents_run[0:max_iters, rmse_col]
                rmse_job.append(rmse_run_temp)

                norm_grad_run_temp = contents_run[0:max_iters, norm_grad_col]
                norm_grad_job.append(norm_grad_run_temp)

                n_runs += 1

        # Save results in a dictionary and update during computation
        self.job_results = {'rmse': np.array(rmse_job), 'norm_grad': np.array(norm_grad_job),
                            'n_runs': n_runs, 'neig_calls': np.array(neig_calls_job), 'max_iters': max_iters}

        # For plotting
        self.do_plot = do_plot

    def filter_outliers(self, rmse_job, n_runs, factor_filter=1.9, ind_iter_filter=[0]):
        """
        Remove outliers from the runs

        We can do this by removing the outliers of the rmse, test error or distance of J from mean
        We choose testing error as this is closest to what we would do in practice

        RMSE is not available in practice because we wouldn't have access to the truth

        Inputs:
            factor_filter:
            ind_iter_filter: The different AL iterations to consider
        """
        for i_iter in ind_iter_filter:
            if self.do_plot:
                # Let's probe the histogram of the first iteration
                plt.figure(1, figsize=(8, 8))
                plt.hist(rmse_job[:, i_iter], bins=int(n_runs / 2))
                plt.xlabel('RMSE')
                plt.ylabel('Count')

            # Remove the outliers using the first iteration
            rmse_mean = np.mean(rmse_job, axis=0)
            rmse_std = np.std(rmse_job, axis=0)

            rmse_runs_iter = np.abs(rmse_job[:, i_iter] - rmse_mean[i_iter])

            outlier_run_ids = np.where(rmse_runs_iter > factor_filter * rmse_std[i_iter])[0].tolist()

            print('Outliers (RUN IDS) in RMSE are:')
            print(outlier_run_ids)

            rmse_job = np.delete(rmse_job, outlier_run_ids, axis=0)

            n_runs = n_runs - len(outlier_run_ids)

        # Get the mean results now
        rmse_mean = np.mean(rmse_job, axis=0)
        rmse_std = np.std(rmse_job, axis=0)

        return rmse_job, n_runs, rmse_mean, rmse_std

    def plot_rmse(self, factor_filter=1.9, ind_iter_filter=[],
                  FLAG_plot=False, label_learners=['Uniform'], save_filename='results.png'):
        """
        Plots the RMSE (as recorded in the log-files) after removing outliers
        """
        rmse_job = self.job_results['rmse']
        n_runs = self.job_results['n_runs']

        if self.FLAG_filter is False:
            ind_iter_filter = []

        rmse_job, n_runs, rmse_mean, rmse_std = self.filter_outliers(rmse_job, n_runs,
                                                                     factor_filter=factor_filter,
                                                                     ind_iter_filter=ind_iter_filter)

        self.job_results['rmse'] = rmse_job
        self.job_results['n_runs'] = n_runs

        if FLAG_plot:
            plotting_options = {'save_filename': save_filename, 'label_learners': label_learners,
                                'FLAG_legend_outside': False}
            plot_trend([rmse_mean], [self.N_p], error_learners=[rmse_std], plotting_options=plotting_options)

        return rmse_mean, rmse_std

    def plot_norm_gradients(self, factor_filter=1.9, ind_iter_filter=[],
                  FLAG_plot=False, label_learners=['Uniform'], save_filename='results.png'):
        """
        Plots the RMSE (as recorded in the log-files) after removing outliers
        """
        rmse_job = self.job_results['norm_grad']
        n_runs = self.job_results['n_runs']

        if self.FLAG_filter is False:
            ind_iter_filter = []

        rmse_job, n_runs, rmse_mean, rmse_std = self.filter_outliers(rmse_job, n_runs,
                                                                     factor_filter=factor_filter,
                                                                     ind_iter_filter=ind_iter_filter)

        self.job_results['rmse'] = rmse_job
        self.job_results['n_runs'] = n_runs

        if FLAG_plot:
            plotting_options = {'save_filename': save_filename, 'label_learners': label_learners,
                                'FLAG_legend_outside': False}
            plot_trend([rmse_mean], [self.N_p], error_learners=[rmse_std], plotting_options=plotting_options)

        return rmse_mean, rmse_std
