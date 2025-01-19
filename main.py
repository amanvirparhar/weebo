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


class Weebo:
    def __init__(self):
        # audio settings
        self.SAMPLE_RATE = 24000
        self.WHISPER_SAMPLE_RATE = 16000
        self.SILENCE_THRESHOLD = 0.02   # volume level that counts as silence
        self.SILENCE_DURATION = 1.5    # seconds of silence before cutting recording

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

    def _init_espeak(self):
        # setup espeak for phoneme generation
        espeak_data_path = espeakng_loader.get_data_path()
        espeak_lib_path = espeakng_loader.get_library_path()
        EspeakWrapper.data_path = espeak_data_path
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
        # state for audio recording
        audio_buffer = []
        silence_frames = 0
        total_frames = 0

        def callback(indata, frames, time_info, status):
            # callback function that processes incoming audio frames
            nonlocal audio_buffer, silence_frames, total_frames

            if status:
                print(status)

            audio = indata.flatten()
            level = np.abs(audio).mean()

            audio_buffer.extend(audio.tolist())
            total_frames += len(audio)

            # track silence duration
            if level < self.SILENCE_THRESHOLD:
                silence_frames += len(audio)
            else:
                silence_frames = 0

            # process audio when silence is detected
            if silence_frames > self.SILENCE_DURATION * self.SAMPLE_RATE:
                audio_segment = np.array(audio_buffer, dtype=np.float32)

                if len(audio_segment) > self.SAMPLE_RATE:
                    text = self.whisper_mlx.transcribe(audio_segment)['text']

                    # skip empty/invalid transcriptions
                    if text.strip():
                        print(f"Transcription: {text}")
                        self.messages.append({
                            'role': 'user',
                            'content': text
                        })
                        self.create_and_play_response(text)

                # reset state
                audio_buffer.clear()
                silence_frames = 0
                total_frames = 0

        # start recording loop
        with sd.InputStream(
            callback=callback,
            channels=1,
            samplerate=self.SAMPLE_RATE,
            dtype=np.float32
        ):
            print("Recording... Press Ctrl+C to stop")
            while True:
                sd.sleep(100)

    def create_and_play_response(self, prompt: str):
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

        # process response stream
        for chunk in stream:
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
        if buffer:
            ph = self.phonemize(buffer)
            futures.append(
                self.executor.submit(
                    self.generate_audio,
                    ph, self.VOICE, self.SPEED
                )
            )

        # play generated audio
        with sd.OutputStream(
            samplerate=self.SAMPLE_RATE,
            channels=1,
            dtype=np.float32
        ) as out_stream:
            for fut in futures:
                audio_data = fut.result()
                if len(audio_data) == 0:
                    continue
                out_stream.write(audio_data.reshape(-1, 1))


def main():
    weebo = Weebo()
    try:
        weebo.record_and_transcribe()
    except KeyboardInterrupt:
        print("Stopping...")


if __name__ == "__main__":
    main()
