import json


class Configurations:
    def __init__(self, config_file):
        config_dict = json.load(config_file)
        self.randomness = config_dict.get("randomness", 0)
        self.shapes = config_dict.get("shapes", {})
        self.num_points = config_dict.get("num_points", 0)


