import yaml

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

def save_config(conf, config_path="config.yaml"):
    with open(config_path, "w") as c:
        yaml.dump(conf, c, default_flow_style=False)