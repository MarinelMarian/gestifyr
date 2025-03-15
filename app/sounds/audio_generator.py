from gtts import gTTS

# List of words
words = ["nod", "shake", "mouth", "eyebrows", "blink", "smile", "none"]

# Generate MP3 files for each word
for word in words:
    tts = gTTS(text=word, lang="en", tld="com", slow=False)
    file_path = f"{word}.wav"
    tts.save(file_path)
    print(f"Saved: {file_path}")
