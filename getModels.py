from TTS.api import TTS

# Initialisiere das TTS-Modell
tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=False)

# Verfügbare Sprecher anzeigen
if tts.speakers:
    print("Verfügbare Sprecher:")
    for idx, speaker in enumerate(tts.speakers):
        print(f"{idx + 1}: {speaker}")
else:
    print("Dieses Modell unterstützt keine spezifischen Sprecher.")
