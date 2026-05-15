import asyncio
import contextlib
import json
import os
import socket
import struct
import tempfile
import wave

from vosk import Model, KaldiRecognizer


HOST = "0.0.0.0"
PORT = 5000
SAMPLE_RATE = 16000

MODEL_PATH = "/home/payson/projects/SpeechServer/vosk_models/vosk-model-en-us-0.42-gigaspeech"
PIPER_BIN = "/home/payson/projects/SpeechServer/home/payson/piper/piper"
PIPER_VOICE = "/home/payson/projects/SpeechServer/home/payson/piper/voices/en_US-amy-medium.onnx"

TTS_SR = 22050

#volume stuff
TTS_GAIN = 0.50

#limiter
PCM_LIMIT = 16000
PADDING_MS = 30

CHUNK_SIZE = 1024

SPEED_FACTOR = 1.75

model = Model(MODEL_PATH)


def wav_to_pcm_s16le_mono(wav_path: str, expected_sr: int) -> bytes:
    with wave.open(wav_path, "rb") as wf:
        channels = wf.getnchannels()
        sample_rate = wf.getframerate()
        sample_width = wf.getsampwidth()
        frame_count = wf.getnframes()
        raw = wf.readframes(frame_count)

    if sample_width != 2:
        raise RuntimeError(f"WAV is not 16-bit PCM. sample_width={sample_width}")

    if sample_rate != expected_sr:
        raise RuntimeError(
            f"WAV sample rate {sample_rate} does not match TTS_SR {expected_sr}. "
            f"Set TTS_SR = {sample_rate} or configure Piper/ESP32 playback to match."
        )

    if channels == 1:
        if len(raw) % 2:
            raw = raw[:-1]
        return raw

    if channels == 2:
        out = bytearray()

        for i in range(0, len(raw) - 3, 4):
            left = struct.unpack_from("<h", raw, i)[0]
            right = struct.unpack_from("<h", raw, i + 2)[0]

            mono = int((left + right) / 2)

            if mono > 32767:
                mono = 32767
            elif mono < -32768:
                mono = -32768

            out += struct.pack("<h", mono)

        return bytes(out)

    raise RuntimeError(f"WAV channels must be 1 or 2. channels={channels}")


def hard_limit_pcm16(pcm: bytes, gain: float = TTS_GAIN, limit: int = PCM_LIMIT) -> bytes:
    if not pcm:
        return b""

    if len(pcm) % 2:
        pcm = pcm[:-1]

    out = bytearray(len(pcm))

    for i in range(0, len(pcm), 2):
        sample = struct.unpack_from("<h", pcm, i)[0]

        # Volume adjustment
        sample = int(sample * gain)

        # Hard limiter
        if sample > limit:
            sample = limit
        elif sample < -limit:
            sample = -limit

        struct.pack_into("<h", out, i, sample)

    return bytes(out)


def add_silence_padding(pcm: bytes, sr: int, ms: int = PADDING_MS) -> bytes:
    if not pcm:
        return b""

    silence_samples = int(sr * ms / 1000)
    silence = b"\x00\x00" * silence_samples
    return silence + pcm + silence


async def piper_synthesize_pcm(text: str) -> bytes:
    text = (text or "").strip()
    if not text:
        return b""

    fd, wav_path = tempfile.mkstemp(suffix=".wav", prefix="piper_")
    os.close(fd)

    try:
        proc = await asyncio.create_subprocess_exec(
            PIPER_BIN,
            "-m",
            PIPER_VOICE,
            "-f",
            wav_path,
            "-q",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        _stdout, stderr = await proc.communicate(
            input=(text + "\n").encode("utf-8", errors="ignore")
        )

        if proc.returncode != 0:
            raise RuntimeError(
                f"piper failed rc={proc.returncode} "
                f"stderr={stderr.decode('utf-8', errors='ignore')}"
            )

        pcm = wav_to_pcm_s16le_mono(wav_path, TTS_SR)

        # Volume + hard limiting
        pcm = hard_limit_pcm16(pcm, gain=TTS_GAIN, limit=PCM_LIMIT)

        # Padding helps avoid pops/blasts at start/end
        pcm = add_silence_padding(pcm, TTS_SR, ms=PADDING_MS)

        if len(pcm) % 2:
            pcm = pcm[:-1]

        return pcm

    finally:
        with contextlib.suppress(Exception):
            os.remove(wav_path)


async def send_json_line(writer: asyncio.StreamWriter, obj: dict):
    writer.write((json.dumps(obj, separators=(",", ":")) + "\n").encode("utf-8"))
    await writer.drain()


async def send_tts_audio(writer: asyncio.StreamWriter, text: str):
    pcm = await piper_synthesize_pcm(text)

    if len(pcm) % 2:
        pcm = pcm[:-1]

    header = {
        "type": "tts_audio",
        "text": text,
        "sr": TTS_SR,
        "fmt": "pcm_s16le",
        "ch": 1,
        "bytes": len(pcm),
    }

    writer.write((json.dumps(header, separators=(",", ":")) + "\n").encode("utf-8"))
    await writer.drain()

    if not pcm:
        return

    # 16-bit mono PCM = 2 bytes per sample
    bytes_per_second = TTS_SR * 2

    # Send slightly faster than real-time playback
    delay_per_chunk = CHUNK_SIZE / bytes_per_second / SPEED_FACTOR

    total = len(pcm)
    sent = 0

    while sent < total:
        chunk = pcm[sent:sent + CHUNK_SIZE]
        writer.write(chunk)
        await writer.drain()

        sent += len(chunk)

        # Pacing prevents overwhelming ESP32 receive/audio buffer
        await asyncio.sleep(delay_per_chunk)


async def handle_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
    addr = writer.get_extra_info("peername")
    print("client connected:", addr)

    # Lower latency for TCP
    sock = writer.get_extra_info("socket")
    if sock is not None:
        with contextlib.suppress(Exception):
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    # Keep buffering reasonable
    transport = writer.transport
    with contextlib.suppress(Exception):
        transport.set_write_buffer_limits(high=128 * 1024, low=32 * 1024)

    rec = KaldiRecognizer(model, SAMPLE_RATE)
    rec.SetWords(False)

    await send_json_line(writer, {"type": "info", "text": "connected"})

    last_partial = ""

    try:
        while True:
            data = await reader.read(4096)

            if not data:
                break

            # Raw PCM16LE audio from ESP32 mic
            if rec.AcceptWaveform(data):
                result = json.loads(rec.Result())
                text = (result.get("text") or "").strip()

                if text:
                    # Clear partial line before final print
                    print(" " * 120, end="\r")
                    print(f"[FINAL] {text}")

                    last_partial = ""

                    try:
                        # Do not send separate final or partial messages to ESP32.
                        # Text is included inside the tts_audio header.
                        await send_tts_audio(writer, text)

                    except Exception as e:
                        msg = f"TTS error: {repr(e)}"
                        print(msg)

                        # Send info only when TTS fails.
                        await send_json_line(writer, {"type": "info", "text": msg})

            else:
                partial_result = json.loads(rec.PartialResult())
                partial = (partial_result.get("partial") or "").strip()

                if partial and partial != last_partial:
                    last_partial = partial
                    print(f"[PARTIAL] {partial}", end="\r", flush=True)

    except Exception as e:
        print("error:", repr(e))

    finally:
        print("\nclient disconnected:", addr)

        with contextlib.suppress(Exception):
            writer.close()
            await writer.wait_closed()


async def main():
    server = await asyncio.start_server(
        handle_client,
        HOST,
        PORT,
        limit=64 * 1024,
    )

    print(
        f"listening on {HOST}:{PORT} "
        f"(vosk SR={SAMPLE_RATE}, tts SR={TTS_SR}, "
        f"gain={TTS_GAIN}, limit={PCM_LIMIT}, padding={PADDING_MS}ms, "
        f"chunk={CHUNK_SIZE}, speed={SPEED_FACTOR}x)"
    )

    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    asyncio.run(main())
