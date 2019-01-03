"""Command-line script to generate the results for the literature review.

TODO:
- Add CLI with argparse
- Add CL arguments to control savepath, dpi and format
"""

import utils as ut
import analysis as anl


if __name__ == '__main__':

    df = ut.load_data_items()
    results_df = ut.load_reported_results_data()
    ut.check_data_items(df)

    # Introduction
    anl.plot_eeg_intro()

    # Rationale
    anl.plot_prisma_diagram()
    anl.make_domain_table(df)
    anl.plot_domain_tree(df)

    # Origin
    anl.plot_type_of_paper(df)
    anl.plot_country(df)
    # anl.plot_countrymap(df)
    anl.plot_domains_per_year(df)

    # Data
    anl.plot_number_subjects_by_domain(df)
    anl.plot_hardware(df)
    anl.plot_number_channels(df)
    anl.plot_data_quantity(df)
    # anl.compute_stats_sampling_rate(df)

    # EEG methodology
    anl.plot_preprocessing_proportions(df)

    # DL methodology
    anl.plot_architectures(df)
    anl.plot_architectures_per_year(df)
    anl.plot_architectures_vs_input(df)
    anl.plot_number_layers(df)
    anl.plot_hyperparams_proportions(df)
    anl.plot_model_inspection_and_table(df)

    # Reporting of results
    anl.plot_performance_metrics(df)
    anl.plot_cross_validation(df)
    anl.plot_intra_inter_per_year(df)
    anl.plot_model_comparison(df)
    anl.plot_reported_results(results_df, data_items_df=df)

    # Reproducibility
    anl.plot_reproducibility_proportions(df)
    anl.make_dataset_table(df)
