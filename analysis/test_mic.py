import sounddevice as sd
import numpy as np


samplerate = 44100

print("Speak now...")


audio = sd.rec(
    int(5 * samplerate),
    samplerate=samplerate,
    channels=1
)

sd.wait()


volume = np.sqrt(
    np.mean(
        audio ** 2
    )
)


print("Volume:", volume)