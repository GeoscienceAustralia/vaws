# simple python code to convert several input files

import pandas as pd
import os

group_string = 'group_name,dist_order,dist_dir,damage_scenario,trigger_collapse_at,patch_dist,set_zone_to_zero,water_ingress_order'
type_string = 'type_name,strength_mean,strength_std,dead_load_mean,dead_load_std,group_name,costing_area'
conn_string = 'conn_name,type_name,zone_loc,edge,x_coord,y_coord'
house_string = 'name,replace_cost,height,cpe_cov,cpe_k,cpe_str_cov,length,width,roof_cols,roof_rows'
damage_string = 'name,surface_area,envelope_repair_rate,envelope_factor_formula_type,envelope_coeff1,envelope_coeff2,envelope_coeff3,internal_repair_rate,internal_factor_formula_type,internal_coeff1,internal_coeff2,internal_coeff3'

dic_file_list = {'conn_group.csv': 'conn_groups.csv',
                 'conn_type.csv': 'conn_types.csv',
                 'connections.csv': 'connections.csv',
                 'house_data.csv': 'house_data.csv',
                 'damage_costing_data.csv': 'damage_costing_data.csv'}

dic_string = {'conn_group.csv': group_string,
              'conn_type.csv': type_string,
              'connections.csv': conn_string,
              'house_data.csv': house_string,
              'damage_costing_data.csv': damage_string}


def change_header(path_, filename, name_string, new_filename):

    col_names = name_string.split(',')
    try:
        tmp = pd.read_csv(os.path.join(path_, filename), names=col_names,
                          skiprows=1)
    except IOError:
        print(path_, filename)
    else:
        tmp.to_csv(os.path.join(path_, new_filename), index=False)


def change_header_by_dir(path_):

    for old_file, new_file in dic_file_list.iteritems():
        change_header(path_, old_file, dic_string[old_file], new_file)


path_ = '/Users/hyeuk/Projects/vaws/data/houses/test_scenario3'

num_list = range(4, 15)
num_list.pop(2)

for i in num_list:
    path_ = './test_scenario' + str(i)
    change_header_by_dir(path_)

# zones.csv

def split_zones(path_):

    one = 'name,area,cpi_alpha,wall_dir'.split(',')
    other = 'name,S,SW,W,NW,N,NE,E,SE'.split(',')

    try:
        tmp = pd.read_csv(os.path.join(path_, 'zones.csv'))
    except IOError:
        print path_
    else:
        os.rename(os.path.join(path_, 'zones.csv'), os.path.join(path_, 'zones.org.csv'))

        df_one = tmp.loc[:, ['zone', 'zone area', 'alph_cpi', 'wall_dir']]
        df_one.to_csv(os.path.join(path_, 'zones.csv'), header=one, index=False)

        df = tmp.loc[:, ['zone', '1', '2', '3', '4', '5', '6', '7', '8']]
        df.to_csv(os.path.join(path_, 'zones_cpe_mean.csv'), header=other, index=False)

        df1 = tmp.loc[:, ['zone', '1.1', '2.1', '3.1', '4.1', '5.1', '6.1', '7.1', '8.1']]
        df1.to_csv(os.path.join(path_, 'zones_cpe_str_mean.csv'), header=other, index=False)

        df2 = tmp.loc[:, ['zone', '1.2', '2.2', '3.2', '4.2', '5.2', '6.2', '7.2', '8.2']]
        df2.to_csv(os.path.join(path_, 'zones_cpe_eave_mean.csv'), header=other, index=False)

        df3 = tmp.loc[:, ['zone', '1.3', '2.3', '3.3', '4.3', '5.3', '6.3', '7.3', '8.3']]
        df3.to_csv(os.path.join(path_, 'zones_edge.csv'), header=other, index=False)


# new structure
from distutils.dir_util import copy_tree
from distutils.file_util import copy_file

# scen_name = 'test_scenario1'


def create_dir(_path):
    try:
        os.makedirs(_path)
    except OSError:
        print('{} exists'.format(_path))

def new_dir_structure(scen_name):

    org_debris = '/Users/hyeuk/Projects/vaws/data/debris/'
    org_gust = '/Users/hyeuk/Projects/vaws/data/gust_envelope_profiles/'
    new_path = '/Users/hyeuk/Projects/vaws/scenarios/'

    org_path = os.path.join('/Users/hyeuk/Projects/vaws/data/houses/',
                                scen_name)

    # make new directory
    new_path_scen = os.path.join(new_path, scen_name)
    create_dir(new_path_scen)
    create_dir(os.path.join(new_path_scen, 'input'))
    create_dir(os.path.join(new_path_scen, 'output'))

    path_house = os.path.join(new_path_scen, 'input/house')
    path_debris = os.path.join(new_path_scen, 'input/debris')
    path_gust = os.path.join(new_path_scen, 'input/gust_envelope_profiles')

    create_dir(path_house)
    create_dir(path_debris)
    create_dir(path_gust)

    copy_tree(org_path, path_house)
    copy_tree(org_debris, path_debris)
    copy_tree(org_gust, path_gust)

    # copy cfg file
    copy_file(os.path.join(new_path, '{}.cfg'.format(scen_name)),
              os.path.join(new_path_scen, '{}.cfg'.format(scen_name)))


# transpose csv
for root, dirs, files in os.walk(".", topdown=False):
    for name in dirs:
        if 'house' in name:
            filename = os.path.join(root, name, 'house_data.csv')
            if os.path.exists(filename):
                a = pd.read_csv(filename)
                a['cpe_str_k'] = a['cpe_k']
                a.transpose().to_csv(filename, header=None)
