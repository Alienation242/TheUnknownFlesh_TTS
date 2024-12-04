import json
import torch
from TTS.api import TTS
import os
import random

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

def synthesize_variations(config, text, language, speaker_wav=None, batch_size=1, base_output_name="output"):
    """Erstelle mehrere Variationen desselben Textes."""
    device = "cuda" if config.get("use_cuda", False) and torch.cuda.is_available() else "cpu"

    # Initialisiere das TTS-Modell
    tts = TTS(model_name=config["tts_model"], progress_bar=False).to(device)

    # Seed setzen, falls in der Konfiguration definiert
    initial_seed = config.get("seed", None)
    if initial_seed is not None:
        torch.manual_seed(initial_seed)
        random.seed(initial_seed)

    # Mehrfache Variationen generieren
    for i in range(batch_size):
        # Optional: Seed für Variation setzen
        if initial_seed is not None:
            variation_seed = initial_seed + i
            torch.manual_seed(variation_seed)
            random.seed(variation_seed)

        # Dateiname für die Ausgabe
        output_file = f"{base_output_name}_{i + 1}.wav"

        print(f"DEBUG: Generiere Variation {i + 1} mit Seed {variation_seed if initial_seed else 'nicht gesetzt'}...")

        # Text in Sprache umwandeln und speichern
        tts.tts_to_file(
            text=text,
            file_path=output_file,
            speaker_wav=speaker_wav,
            language=language
        )
        print(f"Audio gespeichert: {output_file}")

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

    # Benutzerinteraktion für Text
    input_text = input("Bitte gib den Text ein, der gesprochen werden soll: ")

    # Benutzerinteraktion für Batchgröße
    batch_size = input("Wie viele Variationen des Outputs möchtest du generieren? (Standard: 1): ").strip()
    batch_size = int(batch_size) if batch_size.isdigit() else 1

    # Benutzerinteraktion für Basename der Ausgabe
    base_output_name = input("Gib den Basisnamen für die Ausgabedateien ein (Standard: 'output'): ").strip()
    base_output_name = base_output_name if base_output_name else "output"

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

    # Sprachsynthese ausführen
    synthesize_variations(config, input_text, language=language, speaker_wav=speaker_wav, batch_size=batch_size, base_output_name=base_output_name)
