#!/usr/bin/env python3

import time

from pysinewave import SineWave

sinewave = SineWave()
sinewave.set_frequency(440)
sinewave.play()
time.sleep(0.5)
sinewave.set_frequency(275)
time.sleep(0.5)
sinewave.stop()

sinewave.set_frequency(275)
time.sleep(0.5)
sinewave.stop()
sinewave.set_frequency(386)
time.sleep(1)
sinewave.stop()
sinewave.set_frequency(275)
time.sleep(0.5)