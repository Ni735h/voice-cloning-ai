import streamlit as st
import requests
from pathlib import Path


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

API_URL = "http://127.0.0.1:8000"


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="VoiceClone AI",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background-color: #070812;
    }

    .block-container {
        max-width: 1200px;
        padding-top: 40px;
        padding-bottom: 50px;
    }

    h1 {
        text-align: center;
        font-size: 52px !important;
        font-weight: 800 !important;
        margin-bottom: 5px !important;
    }

    .subtitle {
        text-align: center;
        color: #888da3;
        font-size: 16px;
        margin-bottom: 45px;
    }

    textarea {
        background-color: #101225 !important;
        color: #dfe1eb !important;
        border-radius: 15px !important;
    }

    .stButton > button {
        border-radius: 12px;
        min-height: 48px;
        font-size: 16px;
        font-weight: 600;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# TITLE
# ============================================================

st.title("🎙️ VoiceClone AI")

st.markdown(
    '<div class="subtitle">'
    'Create natural AI speech using your own voice.'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# MAIN COLUMNS
# ============================================================

left, right = st.columns(
    2,
    gap="large"
)


# ============================================================
# LEFT — VOICE UPLOAD
# ============================================================

with left:

    st.subheader("🎙️ Voice Sample")

    st.caption(
        "Upload a clean recording of your voice."
    )

    voice_file = st.file_uploader(
        "Upload your voice",
        type=[
            "wav",
            "mp3",
            "m4a",
            "mpeg",
            "mpga"
        ]
    )

    if voice_file is not None:

        st.success(
            f"Voice sample selected: {voice_file.name}"
        )

        st.audio(
            voice_file,
            format=voice_file.type
        )


# ============================================================
# RIGHT — GENERATE SPEECH
# ============================================================

with right:

    st.subheader("✨ Generate Speech")

    st.caption(
        "Enter the text you want your voice to speak."
    )

    text = st.text_area(
        "Enter your text",
        value="",
        placeholder="Type your text here...",
        height=120
    )

    generate_button = st.button(
        "🎙️ Generate AI Voice",
        use_container_width=True
    )


# ============================================================
# GENERATION
# ============================================================

if generate_button:

    # --------------------------------------------------------
    # CHECK VOICE FILE
    # --------------------------------------------------------

    if voice_file is None:

        st.warning(
            "Please upload your voice sample first."
        )

        st.stop()


    # --------------------------------------------------------
    # CHECK TEXT
    # --------------------------------------------------------

    if not text.strip():

        st.warning(
            "Please enter some text."
        )

        st.stop()


    # --------------------------------------------------------
    # GENERATION
    # --------------------------------------------------------

    with st.spinner(
        "🎙️ Cloning your voice and generating speech..."
    ):

        try:

            # =================================================
            # UPLOAD VOICE
            # =================================================

            upload_response = requests.post(
                f"{API_URL}/upload-voice",

                files={
                    "file": (
                        voice_file.name,
                        voice_file.getvalue(),
                        voice_file.type
                    )
                },

                timeout=120
            )


            # =================================================
            # CHECK UPLOAD
            # =================================================

            if upload_response.status_code != 200:

                st.error(
                    "Voice upload failed:\n\n"
                    + upload_response.text
                )

                st.stop()


            # =================================================
            # GENERATE VOICE
            # =================================================

            generate_response = requests.post(
                f"{API_URL}/generate-voice",

                json={
                    "text": text
                },

                timeout=600
            )


        # =====================================================
        # TIMEOUT
        # =====================================================

        except requests.exceptions.Timeout:

            st.error(
                "Generation took too long. "
                "The model is running on CPU."
            )

            st.stop()


        # =====================================================
        # CONNECTION ERROR
        # =====================================================

        except requests.exceptions.ConnectionError:

            st.error(
                "Cannot connect to FastAPI.\n\n"
                "Make sure the backend is running on:\n"
                "http://127.0.0.1:8000"
            )

            st.stop()


        # =====================================================
        # OTHER ERROR
        # =====================================================

        except Exception as e:

            st.error(
                f"Unexpected error: {str(e)}"
            )

            st.stop()


    # ========================================================
    # RESULT
    # ========================================================

    if generate_response.status_code == 200:

        try:

            data = generate_response.json()

        except Exception:

            st.error(
                "Backend returned an invalid response."
            )

            st.stop()


        # ====================================================
        # OUTPUT PATH
        # ====================================================

        output_path = data.get("file")


        if not output_path:

            st.error(
                "Backend did not return an output file."
            )

            st.stop()


        # ====================================================
        # FIX RELATIVE PATH
        # ====================================================

        output_path = Path(output_path)


        if not output_path.is_absolute():

            output_path = PROJECT_ROOT / output_path


        output_path = output_path.resolve()


        # ====================================================
        # CHECK FILE EXISTS
        # ====================================================

        if not output_path.exists():

            st.error(
                "The voice was generated, but the output "
                "audio file could not be found."
            )

            st.code(
                str(output_path)
            )

            st.info(
                "Check that FastAPI actually created this file."
            )

            st.stop()


        # ====================================================
        # AUDIO ANALYSIS
        # ====================================================

        audio_info = data.get(
            "audio_analysis",
            {}
        )


        # ====================================================
        # SUCCESS
        # ====================================================

        st.success(
            "✨ Your AI voice has been generated!"
        )


        # ====================================================
        # GENERATED VOICE
        # ====================================================

        st.subheader(
            "🎧 Generated Voice"
        )


        # Read audio
        with open(
            output_path,
            "rb"
        ) as audio_file:

            audio_bytes = audio_file.read()


        # Audio player
        st.audio(
            audio_bytes,
            format="audio/wav"
        )


        # ====================================================
        # DOWNLOAD
        # ====================================================

        st.download_button(
            label="⬇️ Download Cloned Voice",

            data=audio_bytes,

            file_name="cloned_voice.wav",

            mime="audio/wav",

            use_container_width=True
        )


        # ====================================================
        # VOICE ANALYSIS
        # ====================================================

        st.subheader(
            "🔍 Voice Analysis"
        )


        c1, c2, c3, c4 = st.columns(4)


        # ----------------------------------------------------
        # QUALITY
        # ----------------------------------------------------

        with c1:

            st.metric(
                "Quality",
                f"{audio_info.get('quality_score', 0)}/100"
            )


        # ----------------------------------------------------
        # DURATION
        # ----------------------------------------------------

        with c2:

            st.metric(
                "Duration",
                f"{audio_info.get('final_duration', 0)} sec"
            )


        # ----------------------------------------------------
        # SAMPLE RATE
        # ----------------------------------------------------

        with c3:

            st.metric(
                "Sample Rate",
                f"{audio_info.get('final_sample_rate', 0)} Hz"
            )


        # ----------------------------------------------------
        # SNR
        # ----------------------------------------------------

        with c4:

            st.metric(
                "SNR",
                f"{audio_info.get('snr', 0)} dB"
            )


        # ====================================================
        # DETAILED ANALYSIS
        # ====================================================

        with st.expander(
            "View detailed audio analysis"
        ):

            st.write(
                "Original Sample Rate:",
                audio_info.get(
                    "original_sample_rate",
                    0
                ),
                "Hz"
            )


            st.write(
                "Final Sample Rate:",
                audio_info.get(
                    "final_sample_rate",
                    0
                ),
                "Hz"
            )


            st.write(
                "Original Duration:",
                audio_info.get(
                    "original_duration",
                    0
                ),
                "seconds"
            )


            st.write(
                "Final Duration:",
                audio_info.get(
                    "final_duration",
                    0
                ),
                "seconds"
            )


            st.write(
                "Channels:",
                audio_info.get(
                    "channels",
                    1
                )
            )


            st.write(
                "SNR:",
                audio_info.get(
                    "snr",
                    0
                ),
                "dB"
            )


            st.write(
                "Reference Transcript:",
                audio_info.get(
                    "reference_text",
                    "Not available"
                )
            )


    # ========================================================
    # GENERATION FAILED
    # ========================================================

    else:

        st.error(
            "Voice generation failed:\n\n"
            + generate_response.text
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "VoiceClone AI • Python • FastAPI • Qwen3-TTS"
)
