import yaml
from models import networks # models folder located in same level as train.py, networks.py is in folder 'models'

def _some_outside_function():
    print("ha ha outside function")


class RoadDetector(): #ToDo : Inherit if there is a superclass
    
    def __init__(self, config):
        super(RoadDetector, self).__init__()
        self.config = config
        self.some_param = config['model']['some_param']
        print('debug: ', self.some_param)

    def build(self):
        print("build is called")
        networks.get_generator()

    def call(self):
        print("call is called")


def train(config_file='config/config.yaml'):
    with open(config_file) as f:
        config = yaml.load(f)
        print('[debug] config : ', config)

    trainer = RoadDetector(config)
    trainer.build()
    _some_outside_function()


if __name__ == '__main__':
    import sys
    config_file = sys.argv[1]
    train(config_file)
