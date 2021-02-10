"""
define all loss classes and get function to let them retrieved
"""

class PerceptualLoss(): #ToDo : inherit some superclass

    def __init__(self, input_shape):
        super(PerceptualLoss, self).__init__()

    def call(self):
        print("some call function")


class DiscriminatorLoss:
    def __init__(self):
        self.queue = None
        self.i = 0

    def add(self):
        print("add function is called")


def get_content_loss(config):
    content_loss = PerceptualLoss(config['input_shape'])

    return content_loss


def get_discriminator_loss(config):
    discLoss = DiscriminatorLoss()

    return discLoss

