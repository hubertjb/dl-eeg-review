# dl-eeg-review
Supplementary material for systematic literature review on deep learning and EEG. 

## TODO:
- [X] Add CSV version of spreadsheet
- [ ] Uniformize the data in each column so we can easily make plots
- [ ] Add scripts to produce figures
- [ ] Set up automatic update of figures whenever there is a change to the spreadsheet
- [ ] Find a nice way to display the CSV file
- [ ] Make a blog post version?

## Issues

* How do we include the link? Do we actually want to include links? (They will die at some point.)
* The excluded studies on Google Drive should go in a separate tab (so we don't have to remove them manually in the code).

## Ideas

* We could download the data directly from the Google Sheets using the [API](https://developers.google.com/drive/api/v3/manage-downloads#downloading_google_documents). This way we wouldn't have to manually save and reload the CSV file every time we change the spreadsheet.