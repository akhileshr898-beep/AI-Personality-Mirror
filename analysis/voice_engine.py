import sounddevice as sd
import numpy as np
import time


class VoiceEngine:


    def __init__(self):

        self.start_time = time.time()

        self.speaking_time = 0


        # Laptop microphone

        self.device = 16



    def analyze_voice(self):

        try:

            duration = 3

            samplerate = 44100


            print("Listening...")


            audio = sd.rec(
                int(duration * samplerate),
                samplerate=samplerate,
                channels=1,
                device=self.device
            )


            sd.wait()



            volume = np.sqrt(
                np.mean(
                    audio ** 2
                )
            )


            print(
                "Volume:",
                round(float(volume),4)
            )



            if volume < 0.01:

                return {

                    "active":False,

                    "voice":"SILENCE",

                    "score":0

                }



            self.speaking_time += duration



            score = int(
                min(
                    volume * 1000,
                    100
                )
            )



            return {

                "active":True,

                "voice":"ACTIVE",

                "score":score

            }



        except Exception as e:


            print(
                "VOICE ERROR:",
                e
            )


            return {

                "active":False,

                "voice":"ERROR",

                "score":0

            }