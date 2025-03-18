import sounddevice as sd
import soundfile as sf
import threading
from dotenv import load_dotenv
import os
load_dotenv()
BASE_PATH = os.getenv("BASE_PATH")
APP_REL_PATH = os.getenv("APP_REL_PATH")

def play_wav(filename):
    def audio_thread():
        data, samplerate = sf.read(filename)
        sd.play(data, samplerate, blocking=False, latency='high')
        sd.wait()  # Ensures playback completes

    thread = threading.Thread(target=audio_thread, daemon=True)
    thread.start()

def check_trigger_and_play_sound(probValues, probabilities_thresholds):
    # files = [f"{BASE_PATH}{APP_REL_PATH}sounds/yes_tudor_voice.wav", 
    #          f"{BASE_PATH}{APP_REL_PATH}sounds/no_tudor_voide.wav", 
    #          f"{BASE_PATH}{APP_REL_PATH}sounds/mouth_tudor_voice.wav", 
    #          f"{BASE_PATH}{APP_REL_PATH}sounds/eyebrows_tudor_voice.wav",
    #          f"{BASE_PATH}{APP_REL_PATH}sounds/blink_tudor_voice.wav", 
    #          f"{BASE_PATH}{APP_REL_PATH}sounds/smile_tudor_voice.wav"]
    files = [f"{BASE_PATH}{APP_REL_PATH}sounds/en/nod.wav", 
             f"{BASE_PATH}{APP_REL_PATH}sounds/en/shake.wav", 
             f"{BASE_PATH}{APP_REL_PATH}sounds/en/mouth.wav", 
             f"{BASE_PATH}{APP_REL_PATH}sounds/en/eyebrows.wav",
             f"{BASE_PATH}{APP_REL_PATH}sounds/en/blink.wav", 
             f"{BASE_PATH}{APP_REL_PATH}sounds/en/smile.wav"]
    for i, prob in enumerate(probValues):
        if prob > probabilities_thresholds[i]:
            play_wav(files[i])
            return True
    return False