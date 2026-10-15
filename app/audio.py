import os
import numpy as np
import librosa
import soundfile as sf
import whisper


TARGET_SAMPLE_RATE = 16000

MIN_DURATION = 3
MAX_DURATION = 60


print("Loading Whisper model...")

whisper_model = whisper.load_model("base")

print("Whisper model loaded successfully!")


def load_audio(input_path: str):

    audio, sample_rate = librosa.load(
        input_path,
        sr=None,
        mono=True
    )

    return audio.astype(np.float32), sample_rate


def remove_silence(audio, sample_rate):

    trimmed_audio, _ = librosa.effects.trim(
        audio,
        top_db=35
    )

    return trimmed_audio.astype(np.float32)


def gentle_noise_reduction(audio, sample_rate):

    if len(audio) < sample_rate:

        return audio

    frame_length = 2048
    hop_length = 512

    rms = librosa.feature.rms(
        y=audio,
        frame_length=frame_length,
        hop_length=hop_length
    )[0]

    quiet_count = max(
        1,
        len(rms) // 15
    )

    quiet_frames = np.argsort(rms)[
        :quiet_count
    ]

    stft = librosa.stft(
        audio,
        n_fft=frame_length,
        hop_length=hop_length
    )

    magnitude = np.abs(stft)
    phase = np.angle(stft)

    noise_profile = np.mean(
        magnitude[:, quiet_frames],
        axis=1,
        keepdims=True
    )

    # Gentle reduction.
    # We deliberately avoid aggressive processing
    # because speaker characteristics are important.
    reduction_strength = 0.25

    cleaned_magnitude = np.maximum(
        magnitude -
        reduction_strength * noise_profile,
        0
    )

    cleaned_stft = (
        cleaned_magnitude *
        np.exp(1j * phase)
    )

    cleaned_audio = librosa.istft(
        cleaned_stft,
        hop_length=hop_length,
        length=len(audio)
    )

    return cleaned_audio.astype(np.float32)


def normalize_volume(audio):

    peak = np.max(
        np.abs(audio)
    )

    if peak <= 1e-8:

        return audio

    # Preserve dynamics better than
    # aggressive peak processing.
    target_peak = 0.90

    normalized = (
        audio / peak * target_peak
    )

    return normalized.astype(np.float32)


def convert_sample_rate(
    audio,
    original_sample_rate
):

    if original_sample_rate == TARGET_SAMPLE_RATE:

        return audio.astype(np.float32)

    return librosa.resample(
        audio,
        orig_sr=original_sample_rate,
        target_sr=TARGET_SAMPLE_RATE
    ).astype(np.float32)


def calculate_snr(
    audio,
    sample_rate
):

    frame_length = 2048
    hop_length = 512

    rms = librosa.feature.rms(
        y=audio,
        frame_length=frame_length,
        hop_length=hop_length
    )[0]

    if len(rms) == 0:

        return 0.0

    sorted_rms = np.sort(rms)

    noise_frames = sorted_rms[
        :max(1, len(sorted_rms) // 10)
    ]

    signal_power = np.mean(
        rms ** 2
    )

    noise_power = np.mean(
        noise_frames ** 2
    )

    if noise_power <= 1e-10:

        return 60.0

    snr = 10 * np.log10(
        signal_power / noise_power
    )

    return float(
        max(
            0,
            min(60, snr)
        )
    )


def calculate_quality_score(
    audio,
    sample_rate
):

    duration = (
        len(audio) /
        sample_rate
    )

    snr = calculate_snr(
        audio,
        sample_rate
    )

    rms = np.sqrt(
        np.mean(audio ** 2)
    )

    clipping_ratio = np.mean(
        np.abs(audio) >= 0.99
    )

    score = 100.0

    if duration < 3:

        score -= 30

    elif duration < 5:

        score -= 10

    if snr < 10:

        score -= 35

    elif snr < 20:

        score -= 20

    elif snr < 30:

        score -= 10

    if rms < 0.01:

        score -= 20

    elif rms < 0.02:

        score -= 10

    if clipping_ratio > 0.01:

        score -= 20

    elif clipping_ratio > 0.001:

        score -= 10

    return int(
        max(
            0,
            min(100, score)
        )
    )


def validate_duration(
    audio,
    sample_rate
):

    duration = (
        len(audio) /
        sample_rate
    )

    if duration < MIN_DURATION:

        return False, (
            f"Audio is too short. "
            f"Minimum duration is "
            f"{MIN_DURATION} seconds."
        )

    if duration > MAX_DURATION:

        return False, (
            f"Audio is too long. "
            f"Maximum duration is "
            f"{MAX_DURATION} seconds."
        )

    return True, (
        f"Duration is valid: "
        f"{duration:.2f} seconds."
    )


def detect_language(
    text
):

    if not text.strip():

        return "unknown"

    hindi_characters = 0
    english_characters = 0

    for char in text:

        if "\u0900" <= char <= "\u097F":

            hindi_characters += 1

        elif (
            ("a" <= char.lower() <= "z")
        ):

            english_characters += 1

    if (
        hindi_characters > 0
        and english_characters > 0
    ):

        return "mixed"

    if hindi_characters > 0:

        return "hindi"

    if english_characters > 0:

        return "english"

    return "unknown"


def transcribe_reference_audio(
    audio_path: str
):

    print(
        "Transcribing reference voice..."
    )

    result = whisper_model.transcribe(
        audio_path,
        fp16=False
    )

    text = result[
        "text"
    ].strip()

    detected_language = result.get(
        "language",
        "unknown"
    )

    print(
        "Detected language:",
        detected_language
    )

    print(
        "Reference transcript:",
        text
    )

    return (
        text,
        detected_language
    )


def process_voice(
    input_path: str
):

    os.makedirs(
        "data/voice_samples",
        exist_ok=True
    )

    print(
        "Loading audio..."
    )

    audio, original_sample_rate = (
        load_audio(input_path)
    )

    original_duration = (
        len(audio) /
        original_sample_rate
    )

    print(
        f"Original sample rate: "
        f"{original_sample_rate} Hz"
    )

    print(
        f"Original duration: "
        f"{original_duration:.2f} seconds"
    )

    valid, duration_message = (
        validate_duration(
            audio,
            original_sample_rate
        )
    )

    if not valid:

        raise ValueError(
            duration_message
        )

    print(
        "Removing leading/trailing silence..."
    )

    audio = remove_silence(
        audio,
        original_sample_rate
    )

    print(
        "Applying gentle noise reduction..."
    )

    audio = gentle_noise_reduction(
        audio,
        original_sample_rate
    )

    print(
        "Normalizing volume..."
    )

    audio = normalize_volume(
        audio
    )

    print(
        "Converting to 16 kHz..."
    )

    audio = convert_sample_rate(
        audio,
        original_sample_rate
    )

    processed_path = (
        "data/voice_samples/"
        "processed_uploaded_voice.wav"
    )

    sf.write(
        processed_path,
        audio,
        TARGET_SAMPLE_RATE,
        subtype="PCM_16"
    )

    quality_score = (
        calculate_quality_score(
            audio,
            TARGET_SAMPLE_RATE
        )
    )

    final_duration = (
        len(audio) /
        TARGET_SAMPLE_RATE
    )

    snr = calculate_snr(
        audio,
        TARGET_SAMPLE_RATE
    )

    print(
        "Creating automatic transcript..."
    )

    reference_text, detected_language = (
        transcribe_reference_audio(
            processed_path
        )
    )

    text_language = detect_language(
        reference_text
    )

    result = {

        "processed_path":
            processed_path,

        "original_sample_rate":
            original_sample_rate,

        "final_sample_rate":
            TARGET_SAMPLE_RATE,

        "original_duration":
            round(
                original_duration,
                2
            ),

        "final_duration":
            round(
                final_duration,
                2
            ),

        "channels":
            1,

        "quality_score":
            quality_score,

        "snr":
            round(
                snr,
                2
            ),

        "reference_text":
            reference_text,

        "detected_language":
            detected_language,

        "reference_text_type":
            text_language
    }

    print(
        "Audio processing completed!"
    )

    return result