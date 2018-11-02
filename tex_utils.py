import re
import jinja2
import os
import shutil


def combine_datasets(datasets, subname):
    toBeRemoved = []
    for dataset in datasets:
        if subname in dataset:
            toBeRemoved.append(dataset)
    for dataset in toBeRemoved:
        datasets.remove(dataset)
    if len(toBeRemoved) > 0:
        datasets.add(subname)


def move_element(odict, thekey, newpos):
    odict[thekey] = odict.pop(thekey)
    i = 0
    for key, value in odict.items():
        if key != thekey and i >= newpos:
            odict[key] = odict.pop(key)
        i += 1
    return odict


def get_options_from_file(path):
    """Read a options file.

    The pattern in the file is '%<key>:<value>'
    """
    with open(path) as f:
        content = f.read()
        keys = re.findall(r"%(.+):", content)
        values = re. findall(r":\s*([\w\W]+?)\s*(?:%|$)", content)

    options = dict(zip(keys, values))
    return options

def get_template(template_file):
    """Get a jinja template with latex tags.

    modified from http://eosrei.net/articles/2015/11/latex-templates-python-and-jinja2-generate-pdfs
    """
    latex_jinja_env = jinja2.Environment(
    	block_start_string = '\BLOCK{',
    	block_end_string = '}',
    	variable_start_string = '\VAR{',
    	variable_end_string = '}',
    	comment_start_string = '\#{',
    	comment_end_string = '}',
    	line_statement_prefix = '%%',
    	line_comment_prefix = '%#',
    	trim_blocks = True,
    	autoescape = False,
    	loader = jinja2.FileSystemLoader(os.path.abspath('/'))
    )
    template = latex_jinja_env.get_template(os.path.realpath(template_file))
    return template

def compile_pdf_from_template(template, insert_variables, out_path):
    """Render a template file and compile it to pdf"""

    rendered_template = template.render(**insert_variables)
    build_d = os.path.join(os.path.dirname(os.path.realpath(out_path)), 'tex_build')
    print(build_d)
    if not os.path.exists(build_d):  # create the build directory if not exisiting
        os.makedirs(build_d)

    temp_out = os.path.join(build_d, "tmp")
    with open(temp_out+'.tex', "w") as f:  # saves tex_code to output file
        f.write(rendered_template)

    os.system('pdflatex -output-directory {} {}'.format(build_d, temp_out+'.tex'))
    shutil.copy2(temp_out+".pdf", os.path.relpath(out_path))

    return rendered_template