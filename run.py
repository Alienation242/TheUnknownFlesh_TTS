import json
import torch
from TTS.api import TTS
import os
import random
import re

def load_config(config_path):
    """Lade die Konfigurationsdatei."""
    with open(config_path, "r") as file:
        config = json.load(file)
    return config

def list_languages(model_name):
    """Liste die verfügbaren Sprachen für ein Modell."""
    tts = TTS(model_name=model_name, progress_bar=False)
    languages = tts.languages
    if not languages:
        print("Dieses Modell unterstützt keine spezifischen Sprachen.")
        return None
    print(f"Verfügbare Sprachen: {languages}")
    return languages

def list_speakers(input_folder):
    """Liste verfügbare Sprecher anhand von WAV-Dateien im Eingabeordner."""
    speakers = [
        file for file in os.listdir(input_folder)
        if file.endswith(".wav")
    ]
    if not speakers:
        print(f"Keine WAV-Dateien im Ordner '{input_folder}' gefunden.")
        return []
    print("Verfügbare Sprecher:")
    for idx, speaker in enumerate(speakers, start=1):
        print(f"{idx}: {speaker}")
    return speakers

def synthesize_sentence_variations(config, sentences, language, speaker=None, speaker_wav=None, emotion="Neutral", speed=1.0, base_temp=0.7, batch_size=1):
    """Erstelle mehrere Variationen jedes Satzes mit erweiterten Parametern."""
    os.makedirs("output", exist_ok=True)  # Ordner erstellen, falls er nicht existiert

    device = "cuda" if config.get("use_cuda", False) and torch.cuda.is_available() else "cpu"

    # Initialisiere das TTS-Modell
    tts = TTS(model_name=config["tts_model"], progress_bar=False).to(device)

    # Seed setzen, falls in der Konfiguration definiert
    initial_seed = config.get("seed", None)
    if initial_seed is not None:
        torch.manual_seed(initial_seed)
        random.seed(initial_seed)

    # Generiere Variationen für jeden Satz
    for idx, sentence in enumerate(sentences):
        sentence_prefix = f"{idx + 1:03d}_" + "_".join(re.sub(r'[^\w\s]', '', sentence).split()[:3])  # Nummerieren und ersten drei Wörter für den Dateinamen verwenden
        for i in range(batch_size):
            # Temperatur für die Variation anpassen
            temp = base_temp + i * 0.1
            temp = min(max(temp, 0.7), 1.3)  # Begrenzen der Temperatur auf sinnvolle Werte

            # Dateiname für die Ausgabe
            output_file = f"output/{sentence_prefix}_var{i + 1}.wav"

            print(f"DEBUG: Generiere Satz {idx + 1}, Variation {i + 1} mit Temperatur {temp}...")

            # Text in Sprache umwandeln und speichern
            tts.tts_to_file(
                text=sentence,
                file_path=output_file,
                speaker=speaker,
                speaker_wav=speaker_wav,
                language=language,
                emotion=emotion,
                speed=speed,
                temperature=temp
            )
            print(f"Audio gespeichert: {output_file}")

def split_text_into_sentences(text):
    """Teile einen Text in Sätze auf basierend auf dem Punkt als Delimiter."""
    sentences = [sentence.strip() for sentence in text.split('.') if sentence.strip()]
    return sentences

if __name__ == "__main__":
    config_path = "config.json"
    input_folder = "inputs/"

    # Konfiguration laden
    config = load_config(config_path)

    # Liste der Sprachen anzeigen
    languages = list_languages(config["tts_model"])

    # Benutzerinteraktion für Sprache
    if languages:
        print("Wähle eine Sprache aus der obigen Liste:")
        language = input("Sprache: ").strip()
        if language not in languages:
            print(f"Ungültige Sprache! Verfügbare Sprachen: {languages}")
            exit(1)
    else:
        print("Keine Sprachen verfügbar. Beende.")
        exit(1)

    # Benutzerinteraktion für Textquelle
    print("Wie möchtest du den Text bereitstellen?")
    print("1: Text manuell eingeben")
    print("2: Text aus Datei (inputs/input.txt)")
    text_choice = input("Wähle 1 oder 2: ").strip()

    if text_choice == "1":
        input_text = input("Bitte gib den Text ein, der gesprochen werden soll: ")
    elif text_choice == "2":
        input_file_path = "inputs/input.txt"
        if not os.path.exists(input_file_path):
            print(f"Die Datei {input_file_path} wurde nicht gefunden.")
            exit(1)
        with open(input_file_path, "r", encoding="utf-8") as file:
            input_text = file.read()
    else:
        print("Ungültige Auswahl. Bitte starte das Skript erneut.")
        exit(1)

    # Text in Sätze aufteilen
    sentences = split_text_into_sentences(input_text)
    if not sentences:
        print("Der eingegebene Text enthält keine gültigen Sätze.")
        exit(1)

    # Benutzerinteraktion für Batchgröße
    batch_size = input("Wie viele Variationen jedes Satzes möchtest du generieren? (Standard: 1): ").strip()
    batch_size = int(batch_size) if batch_size.isdigit() else 1

    # Verfügbare Sprecher auflisten
    speakers = list_speakers(input_folder)
    if not speakers:
        exit(1)

    print("Wähle einen Sprecher aus der obigen Liste:")
    speaker_choice = input("Nummer des Sprechers: ").strip()
    if not speaker_choice.isdigit() or int(speaker_choice) < 1 or int(speaker_choice) > len(speakers):
        print("Ungültige Auswahl. Bitte starte das Skript erneut.")
        exit(1)

    # Gewählten Sprecherpfad bestimmen
    speaker_wav = os.path.join(input_folder, speakers[int(speaker_choice) - 1])

    # Liste der Emotionen anzeigen
    emotions = ["Neutral", "Happy", "Sad", "Angry", "Dull"]
    print("Verfügbare Emotionen:")
    for idx, emo in enumerate(emotions, start=1):
        print(f"{idx}: {emo}")

    emotion_choice = input("Wähle eine Emotion durch Eingabe der Zahl: ").strip()
    if not emotion_choice.isdigit() or int(emotion_choice) < 1 or int(emotion_choice) > len(emotions):
        print("Ungültige Auswahl. Bitte starte das Skript erneut.")
        exit(1)

    emotion = emotions[int(emotion_choice) - 1]

    # Benutzerinteraktion für Geschwindigkeit
    speed = input("Gib die Geschwindigkeit ein (Standard: 1.0): ").strip()
    speed = float(speed) if speed.replace('.', '', 1).isdigit() else 1.0

    # Sprachsynthese für jeden Satz ausführen
    synthesize_sentence_variations(
        config,
        sentences,
        language=language,
        speaker_wav=speaker_wav,
        emotion=emotion,
        speed=speed,
        batch_size=batch_size
    )
