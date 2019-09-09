import yaml


income_const = yaml.load(
    open('income.yml'),
    Loader=yaml.FullLoader)