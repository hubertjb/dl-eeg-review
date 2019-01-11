# Data collection table and code for "Deep learning-based EEG analysis: a systematic review"

<p align="center">
<img width="500" alt="Wordcloud" src="img/DL-EEG_WordCloud.png">
</p>

This repository contains the data collection table and code for our systematic review on deep learning and EEG.
The systematic review currently contains 156 journal papers, conference papers and preprints.
Only studies applying deep learning to scalp EEG data were included. 

## Data tables

[Here is the spreadsheet](https://docs.google.com/spreadsheets/d/1smpU0WSlSq-Al2u_QH3djGn68mTuHlth2fNJTrD3wa8/) containing the breakdown of the reviewed papers. This spreadsheet will be maintained and updated overtime and will therefore not reflect exactly the information in the review paper (published on XX) as new papers get added to it. Each row contains the information for a single paper, while the columns are "data items" (e.g., 'domain of application', 'type of architecture', 'number of layers', etc.).

The csv file used to automatically generate all the figures is saved in `/data/data_items.csv`. The csv file is a saved copy of the main spreadsheet and will be erased and replaced everytime a modification is applied to the spreadsheet. 

An additional table containing the reported results of each paper can be found under `/data/reporting_results.csv`.

Standard spreadsheet software (LibreOffice Calc, Microsoft Excel, Google Sheets) can be used to load and browse the CSV files.


## Contributing

We encourage the community to submit new DL-EEG papers to the data collection table (spreadsheet). If your work is already in the data collection table, we would be grateful to have you approve the summary we did of your paper. If you have published work on DL-EEG that is not included in the table, we would be very happy to include your summary in the table. If you see any error in the table we will be happy to correct it.

1. **Modifications and Corrections**
    - **Open an issue** to discuss what you are suggesting to add to/modify in the data collection table, or directly.
    - **Submit a pull request** to propose changes to the data items CSV file. (we will manually apply them to the spreadsheet)
    - **Send us an email** with the suggested modifications. (we will manually apply them to the spreadsheet)

2. **New Paper(s) to Review**
    - **Open an issue** with the title of the paper (and a link would be nice) and we'll add it to the list to be reviewed.
    - **Send us an email** with the title of the paper (and a link would be nice) and we'll add it to the list to be reviewed.

3. **New Reviewed Paper**
    - Download and fill [this template](https://docs.google.com/spreadsheets/d/16CzRFBg340izqtgC1U7QVBvHMRB5z8P98SNKCji7kUU/).
    - **Submit a pull request** with your csv file (i.e. filled template) in the folder _submissions_.
    - or **Send us an email** with the filled template, either a link to a spreadsheet or a csv file attached to the email.

4. **Reviewing a Paper**
    1. **Consult the spreadsheet** for the list "To be reviewed" in the second tab.
    2. **Identify a paper of interest** from the list and review it throughfully to fill all the data items (i.e. columns)
    4. **See previous section** "New Reviewed Paper" so we can update the spreadsheet with your contribution.


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