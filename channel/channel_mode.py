
class ChannelMode:
    """
    SimulationMode definira režime simulacije.
    Attributes:
        Multipath (int): 0 = exclude, 1 = include
        ThermalNoise (int): 0 = exclude, 1 = include
    """

    def __init__(self, multipath=0, thermal_noise=1):
        self.Multipath = multipath
        self.ThermalNoise = thermal_noise

    @property
    def Multipath(self):
        return self._multipath

    @Multipath.setter
    def Multipath(self, value):
        if value in (0, 1):
            self._multipath = value
        else:
            raise ValueError("Multipath mora biti 0 ili 1")

    @property
    def ThermalNoise(self):
        return self._thermal_noise

    @ThermalNoise.setter
    def ThermalNoise(self, value):
        if value in (0, 1):
            self._thermal_noise = value
        else:
            raise ValueError("ThermalNoise mora biti 0 ili 1")

