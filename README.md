# Data collection table and code for "Deep learning-based EEG analysis: a systematic review"

<p align="center">
<img width="500" alt="Wordcloud" src="img/DL-EEG_WordCloud.png">
</p>

This repository contains the data collection table and code for our systematic review on deep learning and EEG.
The systematic review currently contains 156 journal papers, conference papers and preprints.
Only studies applying deep learning to scalp EEG data were included. 

## Data tables

We provide our [data extraction table](https://docs.google.com/spreadsheets/d/1smpU0WSlSq-Al2u_QH3djGn68mTuHlth2fNJTrD3wa8/) containing the summary of the reviewed papers. This table will be maintained and updated over time and might therefore be slightly out-of-sync with the review as new articles are added. Each row contains the information for a single paper, while the columns are "data items" (e.g., 'domain of application', 'type of architecture', 'number of layers', etc.) describing the various characteristics of the studies.

A CSV version of the table is available under `/data/data_items.csv`. This version is used to automatically generate all the figures in the review. The CSV file is updated everytime the main table is modified.

An additional table containing the [reported results](https://docs.google.com/spreadsheets/d/1smpU0WSlSq-Al2u_QH3djGn68mTuHlth2fNJTrD3wa8/edit#gid=1960227030) of each paper is also made available. Its CSV version can be found under `/data/reporting_results.csv`.

Standard spreadsheet software (LibreOffice Calc, Microsoft Excel, Google Sheets) can be used to load and browse the CSV files.


## Contributing

We encourage interested readers to submit new DL-EEG papers to the data collection table so their summary can be shared with the rest of the community. If your work is already in the data collection table, we would be grateful to have you approve the summary we did of your paper. If you have published work on DL-EEG that is not included in the table, we would be very happy to include your summary in the table. If you see any error in the table we will be happy to correct it.

Depending on the kind of contribution you are planning on doing, please see the following steps:

1. **Modifying or correcting existing entries**
    - Option 1: **Open an issue** on Github to discuss what you are suggesting to add to/modify in the data collection table.
    - Option 2: **Submit a pull request** on Github to directly suggest changes to the data items CSV file.
    - Option 3: **Send us an email** with the suggested modifications.

2. **Requesting the inclusion of a new paper**
    - Option 1: **Open an issue** on Github with the title of the paper, the list of authors and the year, as well as a link to an electronic version of the paper.
    - Option 2: **Send us an email** with the information described just above.

3. **Reviewing a new paper**
    - Download and fill [this template](https://docs.google.com/spreadsheets/d/16CzRFBg340izqtgC1U7QVBvHMRB5z8P98SNKCji7kUU/). See Reviewing instructions.
    - Option 1: **Submit a pull request** on Github with your CSV file (i.e., filled template) in the folder _submissions_.
    - Option 2: **Send us an email** with the filled template, either a link to a spreadsheet or a csv file attached to the email.

4. **Reviewing a Paper**
    1. **Consult the spreadsheet** for the list "To be reviewed" in the second tab.
    2. **Identify a paper of interest** from the list and review it throughfully to fill all the data items (i.e. columns)
    4. **See previous section** "New Reviewed Paper" so we can update the spreadsheet with your contribution.


## Reviewing instructions

...

## Producing the figures and results of the review

To produce the figures and results of the review, a Python 3 environment with the packages listed in `requirements.txt` is necessary.
The packages can be installed with:

```
pip install -r requirements.txt
```

`Graphviz` also needs to be installed : see [instructions](https://www.graphviz.org/download/).

Then, run the `generate_results.py` script:

```
cd dl-eeg-review/code/
python generate_results.py
```

This will generate the figures and a log file containing the results under `figs/`.

_* If you are using the data table (i.e. the spreadsheet) and/or information from the review and/or the figures and/or the code to generate (new) figures, please see the **Citation** section._


## Citation

If you are using the data table (i.e. the spreadsheet) and/or information from the review and/or the figures you need to cite the following work:

_coming soon_