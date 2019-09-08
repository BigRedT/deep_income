import wget

from global_constants import income_const
import utils

utils.mkdir_if_not_exist(income_const['download_dir'])
for data_type in income_const['urls']:
    wget.download(
        income_const['urls'][data_type]['url'],
        income_const['download_dir'])