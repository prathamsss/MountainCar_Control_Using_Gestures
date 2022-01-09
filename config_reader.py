import yaml

def read_config(file_path):
    try:
        with open(file_path, 'r') as config_file:
            config_object = yaml.safe_load(config_file)
        return config_object
    except IOError as e:
        print('ERROR OCCURRED : {}'.format(e))
        raise IOError