from analysis.voice_engine import VoiceEngine


voice = VoiceEngine()


while True:

    data = voice.analyze_voice()

    print(data)