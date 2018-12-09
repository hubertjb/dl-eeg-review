# Inspired by: AKuederle
# https://github.com/AKuederle/Py-Tex-automation-example

import pandas as pd
import tex_utils

#papers = pd.read_csv('./paperx.csv')
papers = pd.read_csv('./papers.csv', header=1)

papers = papers.dropna(axis=1, how='all')

domains = set(papers['Domain'].dropna(axis=0, how='all'))

nested_datasets = {}

for domain in domains:
    print('Domain:' + domain)

    sub_papers = papers[papers['Domain'] == domain]
    sub_datasets = sub_papers['Dataset name'].dropna(axis=0, how='all')

    # 1 Paper might use multiple datasets
    l = [dsname.split(";\n") for dsname in sub_datasets]
    sub_datasets = set([item for sublist in l for item in sublist])

    # ========================================
    # Handle exception to make it "prettier".
    # Exception 1: Combining datasets.
    # - BCI Competition Datasets.
    # ========================================
    tex_utils.combine_datasets(sub_datasets, "BCI Competition")
    tex_utils.combine_datasets(sub_datasets, "TUH")

    # Step 3 - Create nested list of publications per dataset for this domain.
    nested_list = {k: list(sub_papers[sub_papers['Dataset name'].str.contains(k)]['Citation']) for k in sub_datasets}

    # ========================================
    # Handle exception to make it "prettier".
    # Exception 2: Datasets used only once.
    # ========================================
    toBeRemoved = []
    others = []
    for dataset in nested_list:
        if len(nested_list[dataset]) < 2:
            print("Dataset: " + dataset + " : " + str(nested_list[dataset]))
            others.append(nested_list[dataset])
            toBeRemoved.append(dataset)

    for dataset in toBeRemoved:
        nested_list.pop(dataset)

    others = [val for sublist in others for val in sublist]
    if len(others) > 0:
        nested_list['Other Datasets'] = others

    #if 'Internal Recordings' in nested_list:
    #    nested_list = move_element(nested_list, "Internal Recordings", 1)
    #if 'Others' in nested_list:
    #    nested_list = move_element(nested_list, "Others", 2)

    # Step 4 - Save the final list of papers per dataset for this domain.
    nested_datasets[domain] = nested_list

print('LaTeX!')
template = tex_utils.get_template('./table_template.tex')
variable_dict = {'datasets': nested_datasets}
texstr = tex_utils.compile_pdf_from_template(template, variable_dict, './dataset_table.pdf')
print('Done!')

print(texstr)