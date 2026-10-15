import os

import torch
import soundfile as sf

from qwen_tts import Qwen3TTSModel

from app.audio import process_voice


MODEL_NAME = (
    "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
)

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

OUTPUT_PATH = (
    "outputs/cloned_voice.wav"
)


print(
    "Loading Qwen3-TTS model..."
)

model = Qwen3TTSModel.from_pretrained(
    MODEL_NAME,
    device_map=DEVICE,
    dtype=torch.float32
)

print(
    "Qwen3-TTS model loaded!"
)


def detect_target_language(text):

    has_hindi = any(
        "\u0900" <= char <= "\u097F"
        for char in text
    )

    has_english = any(
        (
            "a" <= char.lower() <= "z"
        )
        for char in text
    )

    if has_hindi and has_english:

        return "mixed"

    if has_hindi:

        return "hindi"

    if has_english:

        return "english"

    return "unknown"


def generate_english_voice(
    text,
    processed_audio,
    reference_text,
    output_path
):

    print(
        "Generating English voice..."
    )

    wavs, sample_rate = (
        model.generate_voice_clone(

            text=text,

            language="English",

            ref_audio=processed_audio,

            ref_text=reference_text,

            x_vector_only_mode=False,

            # Lower temperature gives us
            # more deterministic generation.
            temperature=0.7,

            top_k=50,

            top_p=0.9,

            repetition_penalty=1.05
        )
    )

    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True
    )

    sf.write(
        output_path,
        wavs[0],
        sample_rate
    )

    return output_path


def generate_cloned_voice(
    text: str,
    ref_audio: str,
    output_path: str = OUTPUT_PATH
):

    if not text.strip():

        raise ValueError(
            "Text cannot be empty."
        )

    print(
        "Processing reference voice..."
    )

    audio_info = process_voice(
        ref_audio
    )

    processed_audio = (
        audio_info[
            "processed_path"
        ]
    )

    reference_text = (
        audio_info[
            "reference_text"
        ]
    )

    target_language = (
        detect_target_language(
            text
        )
    )

    print(
        "Target language:",
        target_language
    )

    print(
        "Reference transcript:"
    )

    print(
        reference_text
    )

    if target_language == "english":

        output_path = (
            generate_english_voice(
                text=text,
                processed_audio=processed_audio,
                reference_text=reference_text,
                output_path=output_path
            )
        )

    elif target_language == "hindi":

        raise ValueError(
            "Hindi generation is not enabled "
            "in the current Qwen3-TTS model. "
            "The next step is to add IndicF5 "
            "as the Hindi voice-cloning engine."
        )

    elif target_language == "mixed":

        raise ValueError(
            "Mixed Hindi + English generation "
            "will be enabled after the Hindi "
            "engine is added."
        )

    else:

        raise ValueError(
            "Could not determine whether the "
            "text is English or Hindi."
        )

    print(
        "Voice generated successfully!"
    )

    print(
        "Saved to:",
        output_path
    )

    return output_path, audio_info