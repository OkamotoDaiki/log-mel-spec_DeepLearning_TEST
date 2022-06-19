import numpy as np

class ConvertHzMel:
    """
    Class for convert frequency to Mel or Mel to frequnecy.
    To calculate, call the melfilterbank fucntion.

    Attributes:
        fo: frequency Parameter. Default setting is 700.
        mel: Definition of the mel scale. Default setting is 1000.
    """
    def __init__(self, fo=700, mel=1000):
        self.fo = fo
        self.mel = mel


    def calc_mo(self):
        """
        Functions for determining dependent parameters of the Mel scale.
        """
        return self.mel / np.log((self.mel / self.fo) + 1.0)


    def hz2mel(self, f):
        """
        Convert Hz to mel.
        """
        mo = self.calc_mo()
        return mo * np.log(f / self.fo + 1.0)
    

    def mel2hz(self, m):
        """
        Convert mel to Hz
        """
        mo = self.calc_mo()
        return self.fo * (np.exp(m / mo) - 1.0)