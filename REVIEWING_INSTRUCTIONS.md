# Reviewing instructions

This document describes how to add new papers to the data collection tables.

There are two tables that are used to produce the figures and results of the review and that need to be updated when adding a new paper: 1) the data items table and 2) the reported results table.

## 1. Filling out the data items table

In this table, each row is a paper and each column is a "data item". The data items are defined and described in the [second tab of the spreadsheet]().

Some data items have two columns. In that case, the first one is often a freer description with more details, whereas the second (which is labeled `<Data item> (clean)` ) is a to-the-point, often categorical, description of that data item. The `clean` column is used to produce the graphs and results of the review.

## 2. Filling out the reported results table

In, the reported results table, rows correspond to one result reported by a study. When adding a new paper, create as many rows as there are individual results to report - this includes the results for the proposed results ad well as for baseline models.

Here is a quick description of each column:

* **Title**: title of the paper
* **Citation**: bibtex entry for that paper.
* **Task**: short description of the task to which the result applies
* **Metric**: performance metric, such as accuracy, f1-score, etc.
* **Model**: type of model to which the result applies. Constructed by concatenating a type of model `arch` (proposed architecture), `dl` (deep learning baseline) or `trad` (traditional pipeline baseline) with a number.
* **Description**: short description of the model.
* **Result**: numerical value reported in the paper. *Note*: please make sure to include results that are presented as percentages as fractions instead (`0.93` accuracy instead of `93%`).
* **Comment**: any additional useful information regarding that results. For example, we indicated when results had to be read approximately from a figure by a comment.  
