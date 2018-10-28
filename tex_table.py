# Inspired by: AKuederle
# https://github.com/AKuederle/Py-Tex-automation-example

import pandas as pd
import tex_utils

papers = pd.read_csv('./paperx.csv')

domains = papers['Domain'].unique()

nested_datasets = {}

for domain in domains:
    sub_papers = papers[papers['Domain'] == domain]
    sub_datasets = sub_papers['Name'].unique()
    nested_list = {k:list(sub_papers[sub_papers['Name'] == k]['Citation']) for k in sub_datasets}

    nested_datasets[domain] = nested_list


template = tex_utils.get_template('./table_template.tex')
variable_dict = {'datasets': nested_datasets}
texstr = tex_utils.compile_pdf_from_template(template, variable_dict, './dataset_table.pdf')

print(texstr)