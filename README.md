# Data collection table and code for "Deep learning-based EEG analysis: a systematic review"

This repository contains the data collection table and code for our systematic review on deep learning and EEG.
The systematic review currently contains 156 journal papers, conference papers and preprints.
Only studies applying deep learning to scalp EEG data were included.

> Roy, Y., Banville, H., Albuquerque, I., Gramfort, A., Falk, T.H., Faubert, J., (2018, December). Deep learning-based EEG analysis: A review of trends from 2010 to 2018. Poster session presented at the Montreal Artificial Intelligence and Neuroscience (MAIN), Montreal, Canada


## Data tables

The data collection table can be found under `/data/data_items.csv`. Each row contains the information for a single paper, while the columns are "data items" (e.g., 'domain of application', 'type of architecture', 'number of layers', etc.). An additional table containing the reported results of each paper can be found under `/data/reporting_results.csv`.

## Producing the figures and results of the review

To produce the figures and results of the review, a Python 3 environment with the packages listed in `requirements.txt` is necessary.
The packages can be installed with:

```
pip install -r requirements.txt
```

`Graphviz` also needs to be installed: see [instructions](https://www.graphviz.org/download/).

Then, run the `generate_results.py` script:

```
cd dl-eeg-review/code/
python generate_results.py
```

This will generate the figures and a log file containing the results under `figs/`.
 
## Contributing

We gladly accept corrections and new submssions to the data collection table. If your work is already in the data collection table, we would be grateful to have you approve the summary we did of your paper. If you have published work on DL-EEG that is not included in the table, we would be very happy to include your summary in the table.

To contribute, please **open an issue** to discuss what you are suggesting to add to/modify in the data collection table, or directly **submit a pull request** with the proposed changes to the CSV file.

## TODO:
- Remove notebooks
- Set up automatic update of figures whenever there is a change to the spreadsheet
- Find a nice way to display the CSV file
- Make a blog post version?

## Tips and tricks
* [Guide](https://scentellegher.github.io/visualization/2018/05/02/custom-fonts-matplotlib.html) to use another font in matplotlib on macOS and Linux

## To validate before pushing new version
* Make sure the reference numbers in the data quantity graphs is correct. If not, the document has to be compiled again, and the `.bbl` file updated in the repo. The `.bbl` file contains the list of citations in the same order as the document as well as the keys.