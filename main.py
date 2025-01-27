import json
import re
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import onnxruntime
import phonemizer
import sounddevice as sd
from phonemizer.backend.espeak.wrapper import EspeakWrapper
import espeakng_loader
from ollama import chat
from lightning_whisper_mlx import LightningWhisperMLX
import signal
from threading import Event
import torch
from scipy.signal import resample

BAD_SENTENCES = [
    "I",
    "you",
    "You're",
    "THANK YOU",
    "www.microsoft.com",
    "The",
    "BANG",
    "Silence.",
    "Sous-titrage Société Radio-Canada",
    "Sous",
    "Sous-",
]

class Weebo:
    def __init__(self):
        # audio settings
        self.SAMPLE_RATE = 24000
        self.WHISPER_SAMPLE_RATE = 16000
        self.AUDIO_CHUNK_DURATION = 1.0  # Analyze 1-second chunks
        self.AUDIO_CHUNK_SIZE = int(self.SAMPLE_RATE * self.AUDIO_CHUNK_DURATION)

        # text-to-speech settings
        self.MAX_PHONEME_LENGTH = 510
        self.CHUNK_SIZE = 300         # size of text chunks for processing
        self.SPEED = 1.2
        self.VOICE = "am_michael"

        # processing things
        self.MAX_THREADS = 1

        # ollama settings
        self.messages = []
        self.SYSTEM_PROMPT = "Give a conversational response to the following statement or question in 1-2 sentences. The response should be natural and engaging, and the length depends on what you have to say."

        # init components
        self._init_espeak()
        self._init_models()
        self.executor = ThreadPoolExecutor(max_workers=self.MAX_THREADS)

        # interrupt handling
        self.shutdown_event = Event()
        signal.signal(signal.SIGINT, self._signal_handler)

    def _signal_handler(self, signum, frame):
        print("\nStopping...")
        self.shutdown_event.set()

    def _init_espeak(self):
        # setup espeak for phoneme generation
        espeak_data_path = espeakng_loader.get_data_path()
        espeak_lib_path = espeakng_loader.get_library_path()
        EspeakWrapper.set_data_path(espeak_data_path)
        EspeakWrapper.set_library(espeak_lib_path)

        # vocab for phoneme tokenization
        self.vocab = self._create_vocab()

    def _init_models(self):
        # init text-to-speech model
        self.tts_session = onnxruntime.InferenceSession(
            "kokoro-v0_19.onnx",
            providers=["CPUExecutionProvider"]
        )

        # load voice profiles
        with open("voices.json") as f:
            self.voices = json.load(f)

        # init speech recognition model
        self.whisper_mlx = LightningWhisperMLX(model="small", batch_size=12)

        # init silero VAD model
        self.vad_model, self.utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False)
        self.get_speech_timestamps, self.save_audio, self.read_audio, self.VAD_iterator, self.collect_chunks = self.utils

    def _create_vocab(self) -> Dict[str, int]:
        # create mapping of characters/phonemes to integer tokens
        chars = ['$'] + list(';:,.!?¡¿—…"«»"" ') + \
            list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz") + \
            list("ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ")
        return {c: i for i, c in enumerate(chars)}

    def phonemize(self, text: str) -> str:
        # clean text and convert to phonemes
        text = re.sub(r"[^\S \n]", " ", text)
        text = re.sub(r"  +", " ", text).strip()
        phonemes = phonemizer.phonemize(
            text,
            "en-us",
            preserve_punctuation=True,
            with_stress=True
        )
        return "".join(p for p in phonemes.replace("r", "ɹ") if p in self.vocab).strip()

    def generate_audio(self, phonemes: str, voice: str, speed: float) -> np.ndarray:
        # convert phonemes to audio using TTS model
        tokens = [self.vocab[p] for p in phonemes if p in self.vocab]
        if not tokens:
            return np.array([], dtype=np.float32)

        tokens = tokens[:self.MAX_PHONEME_LENGTH]
        style = np.array(self.voices[voice], dtype=np.float32)[len(tokens)]

        audio = self.tts_session.run(
            None,
            {
                'tokens': [[0, *tokens, 0]],
                'style': style,
                'speed': np.array([speed], dtype=np.float32)
            }
        )[0]

        return audio

    def record_and_transcribe(self):
        audio_buffer = []
        accumulated_audio = []
        is_speaking = False

        def callback(indata, frames, time_info, status):
            if self.shutdown_event.is_set():
                raise sd.CallbackStop()

            nonlocal audio_buffer, accumulated_audio, is_speaking

            if status:
                print(status)

            audio = indata.flatten()
            audio_buffer.extend(audio.tolist())

            # Process audio in chunks
            if len(audio_buffer) >= self.AUDIO_CHUNK_SIZE:
                audio_chunk = np.array(audio_buffer[:self.AUDIO_CHUNK_SIZE], dtype=np.float32)
                audio_buffer = audio_buffer[self.AUDIO_CHUNK_SIZE:]  # Remove processed chunk from buffer

                # Resample to 16 kHz (required by Silero VAD)
                audio_chunk_resampled = resample(audio_chunk, int(self.WHISPER_SAMPLE_RATE * self.AUDIO_CHUNK_DURATION))

                # Check for voice activity
                speech_timestamps = self.get_speech_timestamps(audio_chunk_resampled, self.vad_model, sampling_rate=self.WHISPER_SAMPLE_RATE)

                if speech_timestamps:
                    if not is_speaking:
                        print("Voice detected, starting transcription...")
                        is_speaking = True

                    # Accumulate audio during speech detection
                    accumulated_audio.extend(audio_chunk_resampled.tolist())
                else:
                    if is_speaking:
                        print("Silence detected, transcribing...")
                        is_speaking = False

                        # Transcribe accumulated audio
                        if accumulated_audio:
                            accumulated_audio_np = np.array(accumulated_audio, dtype=np.float32)
                            text = self.whisper_mlx.transcribe(accumulated_audio_np)['text']

                            # Validate transcription
                            if text.strip():
                                if text.strip() in BAD_SENTENCES:
                                    print('Hallucination detected. Skipping.')
                                elif not re.fullmatch(r"[A-Za-z\s.,!?\'\"-:;()]*", text):
                                    print('Non-english characters detected. Skipping.', text)
                                else:
                                    print(f"Transcription: {text}")
                                    self.messages.append({
                                        'role': 'user',
                                        'content': text
                                    })
                                    self.create_and_play_response(text)

                            # Clear accumulated audio buffer
                            accumulated_audio.clear()

        # Start audio recording loop
        try:
            with sd.InputStream(
                callback=callback,
                channels=1,
                samplerate=self.SAMPLE_RATE,
                dtype=np.float32
            ):
                print("Recording... Press Ctrl+C to stop")
                while not self.shutdown_event.is_set():
                    sd.sleep(100)
        except sd.CallbackStop:
            pass

    def create_and_play_response(self, prompt: str):
        if self.shutdown_event.is_set():
            return

        # stream response from llm
        stream = chat(
            model='llama3.2',
            messages=[{
                'role': 'system',
                'content': self.SYSTEM_PROMPT
            }] + self.messages,
            stream=True,
        )

        # state for processing response
        futures = []
        buffer = ""
        curr_str = ""

        try:
            # process response stream
            for chunk in stream:
                if self.shutdown_event.is_set():
                    break

                print(chunk)
                text = chunk['message']['content']

                if len(text) == 0:
                    self.messages.append({
                        'role': 'assistant',
                        'content': curr_str
                    })
                    curr_str = ""
                    print(self.messages)
                    continue

                buffer += text
                curr_str += text

                # find end of sentence to chunk at
                last_punctuation = max(
                    buffer.rfind('. '),
                    buffer.rfind('? '),
                    buffer.rfind('! ')
                )

                if last_punctuation == -1:
                    continue

                # handle long chunks
                while last_punctuation != -1 and last_punctuation >= self.CHUNK_SIZE:
                    last_punctuation = max(
                        buffer.rfind(', ', 0, last_punctuation),
                        buffer.rfind('; ', 0, last_punctuation),
                        buffer.rfind('— ', 0, last_punctuation)
                    )

                if last_punctuation == -1:
                    last_punctuation = buffer.find(' ', 0, self.CHUNK_SIZE)

                # process chunk
                # convert chunk to audio
                chunk_text = buffer[:last_punctuation + 1]
                ph = self.phonemize(chunk_text)
                futures.append(
                    self.executor.submit(
                        self.generate_audio,
                        ph, self.VOICE, self.SPEED
                    )
                )
                buffer = buffer[last_punctuation + 1:]

            # process final chunk if any
            if buffer and not self.shutdown_event.is_set():
                ph = self.phonemize(buffer)
                futures.append(
                    self.executor.submit(
                        self.generate_audio,
                        ph, self.VOICE, self.SPEED
                    )
                )

            # play generated audio
            if not self.shutdown_event.is_set():
                with sd.OutputStream(
                    samplerate=self.SAMPLE_RATE,
                    channels=1,
                    dtype=np.float32
                ) as out_stream:
                    for fut in futures:
                        if self.shutdown_event.is_set():
                            break
                        audio_data = fut.result()
                        if len(audio_data) == 0:
                            continue
                        out_stream.write(audio_data.reshape(-1, 1))
        except Exception as e:
            if not self.shutdown_event.is_set():
                raise e


def main():
    weebo = Weebo()
    weebo.record_and_transcribe()


if __name__ == "__main__":
    main()
