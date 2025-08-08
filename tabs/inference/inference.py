import os, sys
import gradio as gr
import regex as re
import shutil
import datetime
import json
import torch
import hashlib
from contextlib import suppress
from urllib.parse import urlparse, parse_qs
from yt_dlp import YoutubeDL
from pydub import AudioSegment

from core import (
    run_infer_script,
    run_batch_infer_script,
)

from rvc.lib.utils import format_title
from tabs.settings.sections.restart import stop_infer

now_dir = os.getcwd()
sys.path.append(now_dir)

model_root = os.path.join(now_dir, "logs")
audio_root = os.path.join(now_dir, "assets", "audios")
custom_embedder_root = os.path.join(
    now_dir, "rvc", "models", "embedders", "embedders_custom"
)

PRESETS_DIR = os.path.join(now_dir, "assets", "presets")
FORMANTSHIFT_DIR = os.path.join(now_dir, "assets", "formant_shift")

os.makedirs(custom_embedder_root, exist_ok=True)

custom_embedder_root_relative = os.path.relpath(custom_embedder_root, now_dir)
model_root_relative = os.path.relpath(model_root, now_dir)
audio_root_relative = os.path.relpath(audio_root, now_dir)

sup_audioext = {
    "wav",
    "mp3",
    "flac",
    "ogg",
    "opus",
    "m4a",
    "mp4",
    "aac",
    "alac",
    "wma",
    "aiff",
    "webm",
    "ac3",
}

names = [
    os.path.join(root, file)
    for root, _, files in os.walk(model_root_relative, topdown=False)
    for file in files
    if (
        file.endswith((".pth", ".onnx"))
        and not (file.startswith("G_") or file.startswith("D_"))
    )
]

default_weight = names[0] if names else None

indexes_list = [
    os.path.join(root, name)
    for root, _, files in os.walk(model_root_relative, topdown=False)
    for name in files
    if name.endswith(".index") and "trained" not in name
]

audio_paths = [
    os.path.join(root, name)
    for root, _, files in os.walk(audio_root_relative, topdown=False)
    for name in files
    if name.endswith(tuple(sup_audioext))
    and "_output" not in name
]

custom_embedders = [
    os.path.join(dirpath, dirname)
    for dirpath, dirnames, _ in os.walk(custom_embedder_root_relative)
    for dirname in dirnames
]


def update_sliders(preset):
    with open(
        os.path.join(PRESETS_DIR, f"{preset}.json"), "r", encoding="utf-8"
    ) as json_file:
        values = json.load(json_file)
    return (
        values["pitch"],
        values["filter_radius"],
        values["index_rate"],
        values["rms_mix_rate"],
        values["protect"],
    )


def update_sliders_formant(preset):
    with open(
        os.path.join(FORMANTSHIFT_DIR, f"{preset}.json"), "r", encoding="utf-8"
    ) as json_file:
        values = json.load(json_file)
    return (
        values["formant_qfrency"],
        values["formant_timbre"],
    )


def export_presets(presets, file_path):
    with open(file_path, "w", encoding="utf-8") as json_file:
        json.dump(presets, json_file, ensure_ascii=False, indent=4)


def import_presets(file_path):
    with open(file_path, "r", encoding="utf-8") as json_file:
        presets = json.load(json_file)
    return presets


def get_presets_data(pitch, filter_radius, index_rate, rms_mix_rate, protect):
    return {
        "pitch": pitch,
        "filter_radius": filter_radius,
        "index_rate": index_rate,
        "rms_mix_rate": rms_mix_rate,
        "protect": protect,
    }


def export_presets_button(
    preset_name, pitch, filter_radius, index_rate, rms_mix_rate, protect
):
    if preset_name:
        file_path = os.path.join(PRESETS_DIR, f"{preset_name}.json")
        presets_data = get_presets_data(
            pitch, filter_radius, index_rate, rms_mix_rate, protect
        )
        with open(file_path, "w", encoding="utf-8") as json_file:
            json.dump(presets_data, json_file, ensure_ascii=False, indent=4)
        return "Export successful"
    return "Export cancelled"


def import_presets_button(file_path):
    if file_path:
        imported_presets = import_presets(file_path.name)
        return (
            list(imported_presets.keys()),
            imported_presets,
            "Presets imported successfully!",
        )
    return [], {}, "No file selected for import."


def list_json_files(directory):
    return [f.rsplit(".", 1)[0] for f in os.listdir(directory) if f.endswith(".json")]


def refresh_presets():
    json_files = list_json_files(PRESETS_DIR)
    return gr.update(choices=json_files)


def generate_inference_filename(input_audio_path, voice_model, pitch=0, index_rate=0.5, filter_radius=3, rms_mix_rate=1.0, protect=0.33, f0_method="rmvpe", hop_length=128, output_type="acapella"):
    """Generate proper filename for inference outputs"""
    original_name = os.path.splitext(os.path.basename(input_audio_path))[0]
    original_name = format_title(original_name) 
    
    model_name = os.path.splitext(os.path.basename(voice_model))[0] if voice_model else "unknown_model"
    model_name = format_title(model_name)  
    
    if output_type == "acapella":
        filename = f"{original_name}_{model_name}_acapella.wav"
    else:  
        filename = f"{original_name} ({model_name} Ver).wav"
    
    output_dir = os.path.dirname(input_audio_path)
    return os.path.join(output_dir, filename)


def output_path_fn(input_audio_path, voice_model="", pitch=0, index_rate=0.5, filter_radius=3, rms_mix_rate=1.0, protect=0.33, f0_method="rmvpe", hop_length=128):
    """Generate output path for inference - acapella version by default"""
    if input_audio_path.startswith("http"):
        video_id = get_youtube_video_id(input_audio_path)
        if video_id:
            youtube_dir = os.path.join(audio_root_relative, video_id)
            placeholder_path = os.path.join(youtube_dir, "youtube_audio.wav")
            return generate_inference_filename(placeholder_path, voice_model, pitch, index_rate, filter_radius, rms_mix_rate, protect, f0_method, hop_length, "acapella")
        else:
            return os.path.join(audio_root_relative, "youtube_output.wav")
    else:
        return generate_inference_filename(input_audio_path, voice_model, pitch, index_rate, filter_radius, rms_mix_rate, protect, f0_method, hop_length, "acapella")


def change_choices(model):
    if model:
        speakers = get_speakers_id(model)
    else:
        speakers = [0]
    names = [
        os.path.join(root, file)
        for root, _, files in os.walk(model_root_relative, topdown=False)
        for file in files
        if (
            file.endswith((".pth", ".onnx"))
            and not (file.startswith("G_") or file.startswith("D_"))
        )
    ]

    indexes_list = [
        os.path.join(root, name)
        for root, _, files in os.walk(model_root_relative, topdown=False)
        for name in files
        if name.endswith(".index") and "trained" not in name
    ]

    audio_paths = [
        os.path.join(root, name)
        for root, _, files in os.walk(audio_root_relative, topdown=False)
        for name in files
        if name.endswith(tuple(sup_audioext))
        and "_output" not in name
    ]

    return (
        {"choices": sorted(names), "__type__": "update"},
        {"choices": sorted(indexes_list), "__type__": "update"},
        {"choices": sorted(audio_paths), "__type__": "update"},
        {
            "choices": (
                sorted(speakers)
                if speakers is not None and isinstance(speakers, (list, tuple))
                else [0]
            ),
            "__type__": "update",
        },
        {
            "choices": (
                sorted(speakers)
                if speakers is not None and isinstance(speakers, (list, tuple))
                else [0]
            ),
            "__type__": "update",
        },
    )


def get_indexes():
    indexes_list = [
        os.path.join(dirpath, filename)
        for dirpath, _, filenames in os.walk(model_root_relative)
        for filename in filenames
        if filename.endswith(".index") and "trained" not in filename
    ]

    return indexes_list if indexes_list else ""


def extract_model_and_epoch(path):
    base_name = os.path.basename(path)
    match = re.match(r"(.+?)_(\d+)e_", base_name)
    if match:
        model, epoch = match.groups()
        return model, int(epoch)
    return "", 0


def save_to_wav(record_button):
    if record_button is None:
        pass
    else:
        path_to_file = record_button
        new_name = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".wav"
        target_path = os.path.join(audio_root_relative, os.path.basename(new_name))

        shutil.move(path_to_file, target_path)
        return target_path, output_path_fn(target_path)


def save_to_wav2(upload_audio):
    file_path = upload_audio
    formated_name = format_title(os.path.basename(file_path))
    target_path = os.path.join(audio_root_relative, formated_name)

    if os.path.exists(target_path):
        os.remove(target_path)

    shutil.copy(file_path, target_path)
    return target_path, output_path_fn(target_path)

# function to download audio from youtube
def download_yt_audio(youtube_link):
    """Download audio from a YouTube URL and save it to a organized directory structure."""
    if not youtube_link or not youtube_link.startswith("http"):
        gr.Info("Please provide a valid YouTube URL.")
        return "", ""
    
    video_id = get_youtube_video_id(youtube_link)
    if video_id is None:
        gr.Info("Invalid YouTube URL.")
        return "", ""
    
    song_dir = os.path.join(audio_root_relative, video_id)
    os.makedirs(song_dir, exist_ok=True)
    
    import yt_dlp
    ydl_opts = {
        "format": "bestaudio",
        "outtmpl": os.path.join(song_dir, "%(title)s.%(ext)s"),
        "nocheckcertificate": True,
        "ignoreerrors": True,
        "no_warnings": True,
        "quiet": True,
        "extractaudio": True,
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "wav"}],
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_link, download=True)
            download_path = ydl.prepare_filename(info)
            if not download_path.lower().endswith('.wav') and os.path.exists(download_path):
                base = os.path.splitext(download_path)[0]
                wav_path = f"{base}.wav"
                try:
                    AudioSegment.from_file(download_path).export(wav_path, format="wav")
                    os.remove(download_path)
                except Exception as conv_e:
                    print(f"Conversion to WAV failed: {conv_e}")
                download_path = wav_path
    except Exception as e:
        gr.Info(f"YouTube download error: {e}")
        return "", ""
    rel_path = os.path.relpath(download_path, now_dir)
    return rel_path, output_path_fn(rel_path)

# function to preprocess audio with mdx to separate vocals, instrumentals and reverb
def preprocess_audio_mdx(audio_path, output_path):
    try:
        from mdx import run_mdx
        import json
        mdx_models_dir = os.path.join(now_dir, "rvc", "models", "mdxnet")
        model_data_json = os.path.join(mdx_models_dir, "model_data.json")
        if not (os.path.exists(mdx_models_dir) and os.path.exists(model_data_json)):
            return audio_path, output_path
        with open(model_data_json, "r", encoding="utf-8") as infile:
            mdx_model_params = json.load(infile)
        
        song_dir = os.path.dirname(audio_path)
        if not song_dir:
            song_dir = audio_root_relative
        
        base_name = os.path.splitext(os.path.basename(audio_path))[0]
        instrumentals_path = os.path.join(song_dir, f"{base_name}_Instrumental.wav")
        backup_vocals_path = os.path.join(song_dir, f"{base_name}_Vocals_Backup.wav") 
        main_vocals_dereverb_path = os.path.join(song_dir, f"{base_name}_Vocals_Main_DeReverb.wav")
        
        if (os.path.exists(instrumentals_path) and 
            os.path.exists(backup_vocals_path) and 
            os.path.exists(main_vocals_dereverb_path)):
            print("MDX files already exist, skipping preprocessing...")
            new_output_path = os.path.join(song_dir, os.path.basename(output_path))
            return main_vocals_dereverb_path, new_output_path, instrumentals_path, backup_vocals_path
        
        base_audio_path = os.path.join(now_dir, audio_path)
        vocals_path, instrumentals_path = run_mdx(
            mdx_model_params,
            song_dir,
            os.path.join(mdx_models_dir, "UVR-MDX-NET-Voc_FT.onnx"),
            base_audio_path,
            denoise=True,
            keep_orig=True,
        )
        backup_vocals_path, main_vocals_path = run_mdx(
            mdx_model_params,
            song_dir,
            os.path.join(mdx_models_dir, "UVR_MDXNET_KARA_2.onnx"),
            vocals_path,
            suffix="Backup",
            invert_suffix="Main",
            denoise=True,
        )
        _, main_vocals_dereverb_path = run_mdx(
            mdx_model_params,
            song_dir,
            os.path.join(mdx_models_dir, "Reverb_HQ_By_FoxJoy.onnx"),
            main_vocals_path,
            invert_suffix="DeReverb",
            exclude_main=True,
            denoise=True,
        )
        new_output_path = os.path.join(song_dir, os.path.basename(output_path))
        if os.path.exists(vocals_path):
            os.remove(vocals_path)
        if os.path.exists(main_vocals_path):
            os.remove(main_vocals_path)
        return main_vocals_dereverb_path, new_output_path, instrumentals_path, backup_vocals_path
    except Exception as e:
        print(f"MDX preprocessing error: {e}")
        return audio_path, output_path

def delete_outputs():
    gr.Info(f"Inference outputs cleared!")
    for root, _, files in os.walk(audio_root_relative, topdown=False):
        for name in files:
            if name.endswith(tuple(sup_audioext)) and name.__contains__("_output"):
                os.remove(os.path.join(root, name))


def match_index(model_file_value):
    if model_file_value:
        model_folder = os.path.dirname(model_file_value)
        model_name = os.path.basename(model_file_value)
        index_files = get_indexes()
        pattern = r"^(.*?)_"
        match = re.match(pattern, model_name)
        for index_file in index_files:
            if os.path.dirname(index_file) == model_folder:
                return index_file
            elif match and match.group(1) in os.path.basename(index_file):
                return index_file
            elif model_name in os.path.basename(index_file):
                return index_file
    return ""


def create_folder_and_move_files(folder_name, bin_file, config_file):
    if not folder_name:
        return "Folder name must not be empty."

    folder_name = os.path.basename(folder_name)
    target_folder = os.path.join(custom_embedder_root, folder_name)
    normalized_target_folder = os.path.abspath(target_folder)
    normalized_custom_embedder_root = os.path.abspath(custom_embedder_root)

    if not normalized_target_folder.startswith(normalized_custom_embedder_root):
        return "Invalid folder name. Folder must be within the custom embedder root directory."

    os.makedirs(target_folder, exist_ok=True)

    if bin_file:
        shutil.copy(bin_file, os.path.join(target_folder, os.path.basename(bin_file)))

    if config_file:
        shutil.copy(config_file, os.path.join(target_folder, os.path.basename(config_file)))

    return f"Files moved to folder {target_folder}"


def refresh_formant():
    json_files = list_json_files(FORMANTSHIFT_DIR)
    return gr.update(choices=json_files)


def refresh_embedders_folders():
    custom_embedders = [
        os.path.join(dirpath, dirname)
        for dirpath, dirnames, _ in os.walk(custom_embedder_root_relative)
        for dirname in dirnames
    ]
    return custom_embedders


def get_speakers_id(model):
    if model:
        try:
            model_data = torch.load(os.path.join(now_dir, model), map_location="cpu", weights_only=True)
            speakers_id = model_data.get("speakers_id")
            if speakers_id:
                return list(range(speakers_id))
            else:
                return [0]
        except Exception as e:
            print(f"Error loading model: {e}")
            return [0]
    else:
        return [0]


# Inference tab
def inference_tab():
    with gr.Column():
        with gr.Row():
            model_file = gr.Dropdown(
                label="Voice Model",
                info="Select the voice model to use for inference.",
                choices=sorted(names, key=lambda x: extract_model_and_epoch(x)),
                interactive=True,
                value=default_weight,
                allow_custom_value=True,
            )

            index_file = gr.Dropdown(
                label="Index File",
                info="Select the index file to use for inference.",
                choices=get_indexes(),
                value=match_index(default_weight) if default_weight else "",
                interactive=True,
                allow_custom_value=True,
            )
        with gr.Row():
            unload_button = gr.Button("Unload the voice model")
            refresh_button = gr.Button("Refresh models, indexes and audios")

            unload_button.click(
                fn=lambda: (
                    {"value": "", "__type__": "update"},
                    {"value": "", "__type__": "update"},
                ),
                inputs=[],
                outputs=[model_file, index_file],
            )

            model_file.select(
                fn=lambda model_file_value: match_index(model_file_value),
                inputs=[model_file],
                outputs=[index_file],
            )

    # Single inference tab
    with gr.Tab("Single input infer"):
        with gr.Column():
            upload_audio = gr.Audio(
                label="Upload Audio", type="filepath", editable=False
            )
            with gr.Row():
                audio = gr.Textbox(
                    label="Youtube Audio Input",
                    info="Paste YouTube URL for inference.",
                    placeholder="https://www.youtube.com/watch?v=...",
                    value="",
                    interactive=True,
                )

            audio_dropdown = gr.Dropdown(
                label="Select Local Audio Input",
                info="Select the audio for inference.",
                choices=sorted(audio_paths),
                value="",
                interactive=True,
                allow_custom_value=True,
            )
        has_bg_music = gr.Checkbox(label="Input contains background music", value=False)

        with gr.Accordion("Advanced Settings for inference", open=False):
            with gr.Column():
                clear_outputs_infer = gr.Button("Clear '_output' audio files ( infer outputs ) from 'assets/audios' ")
                output_path = gr.Textbox(
                    label="Path for infer outputs",
                    placeholder="Provide the path for inference outputs",
                    info="The path where inference outputs will be saved. \nBy default they land in 'assets/audios' ",
                    value=(
                        output_path_fn(audio_paths[0])
                        if audio_paths
                        else os.path.join(now_dir, "assets", "audios", "output.wav")
                    ),
                    interactive=True,
                )
                export_format = gr.Radio(
                    label="Export Format",
                    info="Choose the audio export format.",
                    choices=["WAV", "MP3", "FLAC", "OGG", "M4A"],
                    value="WAV",
                    interactive=True,
                )
                sid = gr.Dropdown(
                    label="Speaker ID",
                    info="Select the speaker ID used for inference. \nApplicable only for multi-speaker models.",
                    choices=get_speakers_id(model_file.value),
                    value=0,
                    interactive=True,
                )
                split_audio = gr.Checkbox(
                    label="Audio splitting",
                    info="Splits the audio into chunks ( based on **silence** regions! ). \nCan potentially improve the results.",
                    visible=True,
                    value=False,
                    interactive=True,
                )
                autotune = gr.Checkbox(
                    label="Autotuning",
                    info="Applies the Autotune effect.",
                    visible=True,
                    value=False,
                    interactive=True,
                )
                autotune_strength = gr.Slider(
                    minimum=0,
                    maximum=1,
                    label="Strength of autotuning",
                    info="Autotune effect's strength. \nHigher values snap the pitch more tightly to the chromatic grid.",
                    visible=False,
                    value=1,
                    interactive=True,
                )
                clean_audio = gr.Checkbox(
                    label="Audio cleanup",
                    info="Cleans your audio using noise detection algorithms, preferable for talking / speech audios.",
                    visible=True,
                    value=False,
                    interactive=True,
                )
                clean_strength = gr.Slider(
                    minimum=0,
                    maximum=1,
                    label="Strength of cleaning",
                    info="Set the strenght of cleaning. If you set it too high, the audio might come out muffly or degraded in quality.",
                    visible=False,
                    value=0.3,
                    interactive=True,
                )
                formant_shifting = gr.Checkbox(
                    label="Formant Shifting",
                    info="Enables formant shifting. Useful in situations where your model is a female but input is a male ( and vice-versa ).",
                    value=False,
                    visible=True,
                    interactive=True,
                )
                post_process = gr.Checkbox(
                    label="Post-Processing",
                    info="Various audio effects and processing for the audio output.",
                    value=False,
                    interactive=True,
                )
                with gr.Row(visible=False) as formant_row:
                    formant_preset = gr.Dropdown(
                        label="Browse presets for formant shifting",
                        info="Presets are located in '/assets/formant_shift' folder.",
                        choices=list_json_files(FORMANTSHIFT_DIR),
                        visible=False,
                        interactive=True,
                    )
                    formant_refresh_button = gr.Button(
                        value="Refresh",
                        visible=False,
                    )
                formant_qfrency = gr.Slider(
                    value=1.0,
                    info="Controls the quefrency used for formant shifting. Default is 1.0.",
                    label="Formant Quefrency.",
                    minimum=0.0,
                    maximum=16.0,
                    step=0.1,
                    visible=False,
                    interactive=True,
                )
                formant_timbre = gr.Slider(
                    value=1.0,
                    info="Adjusts timbre characteristics during formant shifting. Default is 1.0.",
                    label="Formant Timbre",
                    minimum=0.0,
                    maximum=16.0,
                    step=0.1,
                    visible=False,
                    interactive=True,
                )
                reverb = gr.Checkbox(
                    label="Reverb",
                    info="Applies reverb to the audio output",
                    value=False,
                    interactive=True,
                    visible=False,
                )
                reverb_room_size = gr.Slider(
                    minimum=0,
                    maximum=1,
                    label="Reverb; Room Size",
                    info="Set the room size of the reverb.",
                    value=0.5,
                    interactive=True,
                    visible=False,
                )

                reverb_damping = gr.Slider(
                    minimum=0,
                    maximum=1,
                    label="Reverb Damping",
                    info="Set the damping of the reverb.",
                    value=0.5,
                    interactive=True,
                    visible=False,
                )

                reverb_wet_gain = gr.Slider(
                    minimum=0,
                    maximum=1,
                    label="Reverb; Wet Gain",
                    info="Set the wet gain of the reverb.",
                    value=0.33,
                    interactive=True,
                    visible=False,
                )

                reverb_dry_gain = gr.Slider(
                    minimum=0,
                    maximum=1,
                    label="Reverb; Dry Gain",
                    info="Set the dry gain of the reverb.",
                    value=0.4,
                    interactive=True,
                    visible=False,
                )

                reverb_width = gr.Slider(
                    minimum=0,
                    maximum=1,
                    label="Reverb; Width",
                    info="Set the width of the reverb.",
                    value=1.0,
                    interactive=True,
                    visible=False,
                )

                reverb_freeze_mode = gr.Slider(
                    minimum=0,
                    maximum=1,
                    label="Reverb; Freeze Mode",
                    info="Set the freeze mode of the reverb.",
                    value=0.0,
                    interactive=True,
                    visible=False,
                )
                pitch_shift = gr.Checkbox(
                    label="Pitch Shift",
                    info="Enable pitch shifting for the audio.",
                    value=False,
                    interactive=True,
                    visible=False,
                )
                pitch_shift_semitones = gr.Slider(
                    minimum=-12,
                    maximum=12,
                    label="Pitch Shift ( Semitones )",
                    info="Set how many semitones to shift the pitch (up or down).",
                    value=0,
                    interactive=True,
                    visible=False,
                )
                limiter = gr.Checkbox(
                    label="Limiter",
                    info="Apply limiter to the audio.",
                    value=False,
                    interactive=True,
                    visible=False,
                )
                limiter_threshold = gr.Slider(
                    minimum=-60,
                    maximum=0,
                    label="Limiter Threshold dB",
                    info="Set the limiter's threshold ( decibels ).",
                    value=-6,
                    interactive=True,
                    visible=False,
                )
                limiter_release_time = gr.Slider(
                    minimum=0.01,
                    maximum=1,
                    label="Limiter Release Time",
                    info="Set the limiter release time.",
                    value=0.05,
                    interactive=True,
                    visible=False,
                )
                gain = gr.Checkbox(
                    label="Gain",
                    info="Apply gain to the audio.",
                    value=False,
                    interactive=True,
                    visible=False,
                )
                gain_db = gr.Slider(
                    minimum=-60,
                    maximum=60,
                    label="Gain dB",
                    info="Set the gain ( decibels ).",
                    value=0,
                    interactive=True,
                    visible=False,
                )
                distortion = gr.Checkbox(
                    label="Distortion",
                    info="Apply distortion to the audio.",
                    value=False,
                    interactive=True,
                    visible=False,
                )
                distortion_gain = gr.Slider(
                    minimum=-60,
                    maximum=60,
                    label="Distortion Gain",
                    info="Set the distortion gain.",
                    value=25,
                    interactive=True,
                    visible=False,
                )
                chorus = gr.Checkbox(
                    label="chorus",
                    info="Apply chorus to the audio.",
                    value=False,
                    interactive=True,
                    visible=False,
                )
                chorus_rate = gr.Slider(
                    minimum=0,
                    maximum=100,
                    label="Chorus Rate Hz",
                    info="Set the chorus rate ( Hertz ).",
                    value=1.0,
                    interactive=True,
                    visible=False,
                )

                chorus_depth = gr.Slider(
                    minimum=0,
                    maximum=1,
                    label="chorus Depth",
                    info="Set the chorus depth.",
                    value=0.25,
                    interactive=True,
                    visible=False,
                )

                chorus_center_delay = gr.Slider(
                    minimum=7,
                    maximum=8,
                    label="chorus Center Delay ms",
                    info="Set the chorus center delay ms.",
                    value=7,
                    interactive=True,
                    visible=False,
                )

                chorus_feedback = gr.Slider(
                    minimum=0,
                    maximum=1,
                    label="chorus Feedback",
                    info="Set the chorus feedback.",
                    value=0.0,
                    interactive=True,
                    visible=False,
                )

                chorus_mix = gr.Slider(
                    minimum=0,
                    maximum=1,
                    label="Chorus Mix",
                    info="Set the chorus mix.",
                    value=0.5,
                    interactive=True,
                    visible=False,
                )
                bitcrush = gr.Checkbox(
                    label="Bitcrush",
                    info="Apply bitcrush to the audio.",
                    value=False,
                    interactive=True,
                    visible=False,
                )
                bitcrush_bit_depth = gr.Slider(
                    minimum=1,
                    maximum=32,
                    label="Bitcrush Bit Depth",
                    info="Set the bitcrush bit depth.",
                    value=8,
                    interactive=True,
                    visible=False,
                )
                clipping = gr.Checkbox(
                    label="Clipping",
                    info="Apply clipping to the audio.",
                    value=False,
                    interactive=True,
                    visible=False,
                )
                clipping_threshold = gr.Slider(
                    minimum=-60,
                    maximum=0,
                    label="Clipping Threshold",
                    info="Set the clipping threshold.",
                    value=-6,
                    interactive=True,
                    visible=False,
                )
                compressor = gr.Checkbox(
                    label="Compressor",
                    info="Apply compressor to the audio.",
                    value=False,
                    interactive=True,
                    visible=False,
                )
                compressor_threshold = gr.Slider(
                    minimum=-60,
                    maximum=0,
                    label="Compressor Threshold dB",
                    info="Set the compressor threshold dB.",
                    value=0,
                    interactive=True,
                    visible=False,
                )

                compressor_ratio = gr.Slider(
                    minimum=1,
                    maximum=20,
                    label="Compressor Ratio",
                    info="Set the compressor ratio.",
                    value=1,
                    interactive=True,
                    visible=False,
                )

                compressor_attack = gr.Slider(
                    minimum=0.0,
                    maximum=100,
                    label="Compressor Attack ms",
                    info="Set the compressor attack ms.",
                    value=1.0,
                    interactive=True,
                    visible=False,
                )

                compressor_release = gr.Slider(
                    minimum=0.01,
                    maximum=100,
                    label="Compressor Release ms",
                    info="Set the compressor release ms.",
                    value=100,
                    interactive=True,
                    visible=False,
                )
                delay = gr.Checkbox(
                    label="Delay",
                    info="Apply delay to the audio.",
                    value=False,
                    interactive=True,
                    visible=False,
                )
                delay_seconds = gr.Slider(
                    minimum=0.0,
                    maximum=5.0,
                    label="Delay Seconds",
                    info="Set the delay seconds.",
                    value=0.5,
                    interactive=True,
                    visible=False,
                )

                delay_feedback = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    label="Delay Feedback",
                    info="Set the delay feedback.",
                    value=0.0,
                    interactive=True,
                    visible=False,
                )

                delay_mix = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    label="Delay Mix",
                    info="Set the delay mix.",
                    value=0.5,
                    interactive=True,
                    visible=False,
                )
                with gr.Accordion("Preset Settings", open=False):
                    with gr.Row():
                        preset_dropdown = gr.Dropdown(
                            label="Select Custom Preset",
                            choices=list_json_files(PRESETS_DIR),
                            interactive=True,
                        )
                        presets_refresh_button = gr.Button("Refresh Presets")
                    import_file = gr.File(
                        label="Select file to import",
                        file_count="single",
                        type="filepath",
                        interactive=True,
                    )
                    import_file.change(
                        import_presets_button,
                        inputs=import_file,
                        outputs=[preset_dropdown],
                    )
                    presets_refresh_button.click(
                        refresh_presets, outputs=preset_dropdown
                    )
                    with gr.Row():
                        preset_name_input = gr.Textbox(
                            label="Preset Name",
                            placeholder="Enter preset name",
                        )
                        export_button = gr.Button("Export Preset")
                pitch = gr.Slider(
                    minimum=-24,
                    maximum=24,
                    step=1,
                    label="Pitch",
                    info="Set the pitch of the audio, the higher the value, the higher the pitch. \nCheat-sheet: 0 = 1:1 as input, 12 = 1 octave higher, -12 = 1 octave lower. \n ***You can also try: 6, 3, -3, -6. Some singers do this if they're uncomfortable with the song's tonality or vocal range.***",
                    value=0,
                    interactive=True,
                )
                filter_radius = gr.Slider(
                    minimum=0,
                    maximum=1,
                    label="Filter Radius / 'threshold' for FCPE",
                    info="f0 smoothing / pitch contour filtering \n-Lower values preserve more of the natural variations in the pitch, including subtle pitch shifts and fluctuations. \n( More dynamic, expressive pitch that might better capture natural pitch variation but could also introduce more 'noise' or instability. ) \n \n-Higher values remove more of the fine details and fluctuations in the pitch, resulting in a smoother and more stable pitch curve. \n ( Yet, potentially flatter and innatural sounding sound + loss of fine-details. ) \n \n ( Best to leave it set to the default '0.006', especially if you're unsure of how it works. ",
                    value=0.006,
                    step=0.001,
                    interactive=False,
                    visible=False,
                )
                index_rate = gr.Slider(
                    minimum=0,
                    maximum=1,
                    label="Search Feature Ratio",
                    info="Influence exerted by the index file; a higher value corresponds to greater influence. However, opting for lower values can help mitigate artifacts present in the audio. \n ***Basically, worse models can't afford to have it too high else you'll get potential artifacts.***",
                    value=0.5,
                    interactive=True,
                )
                rms_mix_rate = gr.Slider(
                    minimum=0,
                    maximum=1,
                    label="RMS Volume Envelope",
                    info="Adjust the loudness (RMS) of the converted voice to match the original/input voice. \n At 1, the output stays the same; values closer to 0 make the output match the input's loudness more closely. \n ***Recommended to leave it at 1, it's a pretty crap functionality.***",
                    value=1,
                    interactive=True,
                )
                protect = gr.Slider(
                    minimum=0,
                    maximum=0.5,
                    label="Protect Voiceless Consonants",
                    info="Safeguard distinct consonants and breathing sounds to prevent electric / buzz, tearing and other artifacts. \n Setting it to its max value of 0.5 offers comprehensive protection.\n ***Generally speaking, higher it is potentially lower the index accuracy.***",
                    value=0.33,
                    interactive=True,
                )
                preset_dropdown.change(
                    update_sliders,
                    inputs=preset_dropdown,
                    outputs=[
                        pitch,
                        filter_radius,
                        index_rate,
                        rms_mix_rate,
                        protect,
                    ],
                )
                export_button.click(
                    export_presets_button,
                    inputs=[
                        preset_name_input,
                        pitch,
                        filter_radius,
                        index_rate,
                        rms_mix_rate,
                        protect,
                    ],
                )
                hop_length = gr.Slider(
                    minimum=1,
                    maximum=512,
                    step=1,
                    label="Hop Length",
                    info="Denotes the duration it takes for the system to transition to a significant pitch change. Smaller hop lengths require more time for inference but tend to yield higher pitch accuracy.",
                    visible=False,
                    value=128,
                    interactive=True,
                )
                f0_method = gr.Radio(
                    label="Pitch extraction algorithm",
                    info="Pitch extraction algorithm to use for the audio conversion. The default algorithm is rmvpe, which is recommended for most cases.",
                    choices=[
                        "crepe",
                        "crepe-tiny",
                        "rmvpe",
                        "fcpe",
                    ],
                    value="rmvpe",
                    interactive=True,
                )
                embedder_model = gr.Radio(
                    label="Embedder Model",
                    info="Model used for learning speaker embedding.",
                    choices=[
                        "contentvec",
                        "spin",
                        "chinese-hubert-base",
                        "japanese-hubert-base",
                        "korean-hubert-base",
                        "custom",
                    ],
                    value="contentvec",
                    interactive=True,
                )
                with gr.Column(visible=False) as embedder_custom:
                    with gr.Accordion("Custom Embedder", open=True):
                        with gr.Row():
                            embedder_model_custom = gr.Dropdown(
                                label="Select Custom Embedder",
                                choices=refresh_embedders_folders(),
                                interactive=True,
                                allow_custom_value=True,
                            )
                            refresh_embedders_button = gr.Button("Refresh embedders")
                        folder_name_input = gr.Textbox(label="Folder Name", interactive=True)
                        with gr.Row():
                            bin_file_upload = gr.File(
                                label="Upload .bin",
                                type="filepath",
                                interactive=True,
                            )
                            config_file_upload = gr.File(
                                label="Upload .json",
                                type="filepath",
                                interactive=True,
                            )
                        move_files_button = gr.Button("Move files to custom embedder folder")

                f0_file = gr.File(
                    label="The f0 curve represents the variations in the base frequency of a voice over time, showing how pitch rises and falls.",
                    visible=True,
                )

        def enforce_terms(terms_accepted, *args):
            if not terms_accepted:
                message = "You must agree to the Terms of Use to proceed."
                gr.Info(message)
                return message, None
            args = list(args)
            
            try:
                pitch = args[0] if len(args) > 0 else 0
                filter_radius = args[1] if len(args) > 1 else 3
                index_rate = args[2] if len(args) > 2 else 0.5
                rms_mix_rate = args[3] if len(args) > 3 else 1.0
                protect = args[4] if len(args) > 4 else 0.33
                hop_length = args[5] if len(args) > 5 else 128
                f0_method = args[6] if len(args) > 6 else "rmvpe"
                original_audio_path = args[7] if len(args) > 7 else ""
                
                if not original_audio_path:
                    message = "No audio input provided."
                    gr.Info(message)
                    return message, None
                has_bg_music_flag = args[8] if len(args) > 8 else False
                original_output_path = args[9] if len(args) > 9 else ""
                model_file = args[10] if len(args) > 10 else ""
                
                if not model_file:
                    message = "No model path provided."
                    gr.Info(message)
                    return message, None
                
                if original_audio_path.startswith("http"):
                    video_id = get_youtube_video_id(original_audio_path)
                    if video_id is None:
                        gr.Info("Invalid YouTube URL.")
                        return "Invalid YouTube URL.", None
                    
                    song_dir = os.path.join(audio_root_relative, video_id)
                    os.makedirs(song_dir, exist_ok=True)
                    
                    # Check for existing audio files
                    existing_audio = None
                    for ext in ['wav', 'mp3', 'mp4', 'm4a', 'flac', 'ogg']:
                        pattern = os.path.join(song_dir, f"*.{ext}")
                        import glob
                        matching_files = glob.glob(pattern)
                        if matching_files:
                            existing_audio = matching_files[0]
                            break
                    
                    if existing_audio:
                        gr.Info("Audio file already exists, skipping download...")
                        audio_path = os.path.relpath(existing_audio, now_dir)
                    else:
                        gr.Info("Downloading from YouTube...")
                        import yt_dlp
                        ydl_opts = {
                            "format": "bestaudio",
                            "outtmpl": os.path.join(song_dir, "%(title)s.%(ext)s"),
                            "nocheckcertificate": True,
                            "ignoreerrors": True,
                            "no_warnings": True,
                            "quiet": True,
                            "extractaudio": True,
                            "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "wav"}],
                        }
                        
                        try:
                            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                                info = ydl.extract_info(original_audio_path, download=True)
                                download_path = ydl.prepare_filename(info)
                                if not download_path.lower().endswith('.wav') and os.path.exists(download_path):
                                    base = os.path.splitext(download_path)[0]
                                    wav_path = f"{base}.wav"
                                    try:
                                        AudioSegment.from_file(download_path).export(wav_path, format="wav")
                                        os.remove(download_path)
                                    except Exception as conv_e:
                                        print(f"Conversion to WAV failed: {conv_e}")
                                    download_path = wav_path
                            audio_path = os.path.relpath(download_path, now_dir)
                        except Exception as e:
                            gr.Info(f"YouTube download error: {e}")
                            return f"YouTube download error: {e}", None
                
                else:
                    if os.path.dirname(original_audio_path).split(os.sep)[-1] != os.path.basename(audio_root_relative):
                        audio_path = original_audio_path
                    else:
                        full_audio_path = os.path.join(now_dir, original_audio_path)
                        if os.path.exists(full_audio_path):
                            file_hash = get_hash(full_audio_path)
                            
                            song_dir = os.path.join(audio_root_relative, file_hash)
                            os.makedirs(song_dir, exist_ok=True)
                            
                            filename = os.path.basename(original_audio_path)
                            new_audio_path = os.path.join(song_dir, filename)
                            full_new_audio_path = os.path.join(now_dir, new_audio_path)
                            
                            if not os.path.exists(full_new_audio_path):
                                shutil.copy2(full_audio_path, full_new_audio_path)
                            
                            audio_path = new_audio_path
                        else:
                            audio_path = original_audio_path
                
                acapella_output_path = generate_inference_filename(
                    audio_path, model_file, pitch, index_rate, filter_radius, 
                    rms_mix_rate, protect, f0_method, 128, "acapella"
                )
                
                instrumental_path = backup_vocals_path = None
                if has_bg_music_flag:
                    mdx_result = preprocess_audio_mdx(audio_path, acapella_output_path)
                    if len(mdx_result) == 4:
                        new_audio_path, new_output_path, instrumental_path, backup_vocals_path = mdx_result
                        audio_path = new_audio_path
                        acapella_output_path = new_output_path
                        
            except Exception as e:
                print(f"Audio preprocessing error: {e}")
                instrumental_path, backup_vocals_path = None, None
                acapella_output_path = original_output_path
                
            message, ai_vocals_path = run_infer_script(
                pitch=pitch,
                filter_radius=filter_radius, 
                index_rate=index_rate,
                volume_envelope=rms_mix_rate,
                protect=protect,
                hop_length=hop_length,
                f0_method=f0_method,
                input_path=audio_path,
                output_path=acapella_output_path,
                pth_path=model_file,
                index_path=args[11] if len(args) > 11 else "",
                split_audio=args[12] if len(args) > 12 else False,
                f0_autotune=args[13] if len(args) > 13 else False,
                f0_autotune_strength=args[14] if len(args) > 14 else 1.0,
                clean_audio=args[15] if len(args) > 15 else False,
                clean_strength=args[16] if len(args) > 16 else 0.7,
                export_format=str(args[17]) if len(args) > 17 else "WAV",
                f0_file=args[18] if len(args) > 18 else None,
                embedder_model=args[19] if len(args) > 19 else "contentvec",
                embedder_model_custom=args[20] if len(args) > 20 else None
            )
            final_output_path = ai_vocals_path
            
            try:
                if instrumental_path and backup_vocals_path and os.path.exists(ai_vocals_path):
                    cover_output_path = generate_inference_filename(
                        audio_path, model_file, pitch, index_rate, filter_radius,
                        rms_mix_rate, protect, f0_method, 128, "cover"
                    )
                    
                    # Mix stems
                    main_vocal_audio = AudioSegment.from_file(ai_vocals_path) - 4
                    backup_vocal_audio = AudioSegment.from_wav(os.path.join(now_dir, backup_vocals_path)) - 6
                    instrumental_audio = AudioSegment.from_wav(os.path.join(now_dir, instrumental_path)) - 7
                    mixed = instrumental_audio.overlay(backup_vocal_audio).overlay(main_vocal_audio)
                    
                    mixed.export(cover_output_path, format="wav")
                    final_output_path = cover_output_path
                    gr.Info(f"Cover version saved as: {os.path.basename(cover_output_path)}")
                    
            except Exception as e:
                print(f"Error creating cover version: {e}")
                
            return message, final_output_path

        def enforce_terms_batch(terms_accepted, *args):
            if not terms_accepted:
                message = "You must agree to the Terms of Use to proceed."
                gr.Info(message)
                return message, None
            return run_batch_infer_script(*args)

        terms_checkbox = gr.Checkbox(
            label="I agree to the terms of use",
            info="Please ensure compliance with the terms and conditions detailed in [this document](https://github.com/codename0og/codename-rvc-fork-3/blob/main/TERMS_OF_USE.md) before proceeding with your inference.",
            value=False,
            interactive=True,
        )

        convert_button1 = gr.Button("Convert")

        with gr.Row():
            vc_output1 = gr.Textbox(
                label="Output Information",
                info="The output information will be displayed here.",
            )
            vc_output2 = gr.Audio("Export Audio")

    # Batch inference tab
    with gr.Tab("Batch"):
        with gr.Row():
            with gr.Column():
                input_folder_batch = gr.Textbox(
                    label="Input Folder",
                    info="Select the folder containing the audios to convert.",
                    placeholder="Enter input path",
                    value=os.path.join(now_dir, "assets", "audios"),
                    interactive=True,
                )
                output_folder_batch = gr.Textbox(
                    label="Output Folder",
                    info="Select the folder where the output audios will be saved.",
                    placeholder="Enter output path",
                    value=os.path.join(now_dir, "assets", "audios"),
                    interactive=True,
                )
        with gr.Accordion("Advanced Settings", open=False):
            with gr.Column():
                clear_outputs_batch = gr.Button("Clear Outputs (Deletes all audios in assets/audios)")
                export_format_batch = gr.Radio(
                    label="Export Format",
                    info="Select the format to export the audio.",
                    choices=["WAV", "MP3", "FLAC", "OGG", "M4A"],
                    value="WAV",
                    interactive=True,
                )
                sid_batch = gr.Dropdown(
                    label="Speaker ID",
                    info="Select the speaker ID to use for the conversion.",
                    choices=get_speakers_id(model_file.value),
                    value=0,
                    interactive=True,
                )
                split_audio_batch = gr.Checkbox(
                    label="Split Audio",
                    info="Split the audio into chunks for inference to obtain better results in some cases.",
                    visible=True,
                    value=False,
                    interactive=True,
                )
                autotune_batch = gr.Checkbox(
                    label="Autotune",
                    info="Apply a soft autotune to your inferences, recommended for singing conversions.",
                    visible=True,
                    value=False,
                    interactive=True,
                )
                autotune_strength_batch = gr.Slider(
                    minimum=0,
                    maximum=1,
                    label="Autotune Strength",
                    info="Set the autotune strength - the more you increase it the more it will snap to the chromatic grid.",
                    visible=False,
                    value=1,
                    interactive=True,
                )
                clean_audio_batch = gr.Checkbox(
                    label="Clean Audio",
                    info="Clean your audio output using noise detection algorithms, recommended for speaking audios.",
                    visible=True,
                    value=False,
                    interactive=True,
                )
                clean_strength_batch = gr.Slider(
                    minimum=0,
                    maximum=1,
                    label="Clean Strength",
                    info="Set the clean-up level to the audio you want, the more you increase it the more it will clean up, but it is possible that the audio will be more compressed.",
                    visible=False,
                    value=0.5,
                    interactive=True,
                )
                formant_shifting_batch = gr.Checkbox(
                    label="Formant Shifting",
                    info="Enable formant shifting. Used for male to female and vice-versa convertions.",
                    value=False,
                    visible=True,
                    interactive=True,
                )
                post_process_batch = gr.Checkbox(
                    label="Post-Process",
                    info="Post-process the audio to apply effects to the output.",
                    value=False,
                    interactive=True,
                )
                with gr.Row(visible=False) as formant_row_batch:
                    formant_preset_batch = gr.Dropdown(
                        label="Browse presets for formanting",
                        info="Presets are located in /assets/formant_shift folder",
                        choices=list_json_files(FORMANTSHIFT_DIR),
                        visible=False,
                        interactive=True,
                    )
                    formant_refresh_button_batch = gr.Button(
                        value="Refresh",
                        visible=False,
                    )
                formant_qfrency_batch = gr.Slider(
                    value=1.0,
                    info="Default value is 1.0",
                    label="Quefrency for formant shifting",
                    minimum=0.0,
                    maximum=16.0,
                    step=0.1,
                    visible=False,
                    interactive=True,
                )
                formant_timbre_batch = gr.Slider(
                    value=1.0,
                    info="Default value is 1.0",
                    label="Timbre for formant shifting",
                    minimum=0.0,
                    maximum=16.0,
                    step=0.1,
                    visible=False,
                    interactive=True,
                )
                reverb_batch = gr.Checkbox(
                    label="Reverb",
                    info="Apply reverb to the audio.",
                    value=False,
                    interactive=True,
                    visible=False,
                )
                reverb_room_size_batch = gr.Slider(
                    minimum=0,
                    maximum=1,
                    label="Reverb Room Size",
                    info="Set the room size of the reverb.",
                    value=0.5,
                    interactive=True,
                    visible=False,
                )

                reverb_damping_batch = gr.Slider(
                    minimum=0,
                    maximum=1,
                    label="Reverb Damping",
                    info="Set the damping of the reverb.",
                    value=0.5,
                    interactive=True,
                    visible=False,
                )

                reverb_wet_gain_batch = gr.Slider(
                    minimum=0,
                    maximum=1,
                    label="Reverb Wet Gain",
                    info="Set the wet gain of the reverb.",
                    value=0.33,
                    interactive=True,
                    visible=False,
                )

                reverb_dry_gain_batch = gr.Slider(
                    minimum=0,
                    maximum=1,
                    label="Reverb Dry Gain",
                    info="Set the dry gain of the reverb.",
                    value=0.4,
                    interactive=True,
                    visible=False,
                )

                reverb_width_batch = gr.Slider(
                    minimum=0,
                    maximum=1,
                    label="Reverb Width",
                    info="Set the width of the reverb.",
                    value=1.0,
                    interactive=True,
                    visible=False,
                )

                reverb_freeze_mode_batch = gr.Slider(
                    minimum=0,
                    maximum=1,
                    label="Reverb Freeze Mode",
                    info="Set the freeze mode of the reverb.",
                    value=0.0,
                    interactive=True,
                    visible=False,
                )
                pitch_shift_batch = gr.Checkbox(
                    label="Pitch Shift",
                    info="Apply pitch shift to the audio.",
                    value=False,
                    interactive=True,
                    visible=False,
                )
                pitch_shift_semitones_batch = gr.Slider(
                    minimum=-12,
                    maximum=12,
                    label="Pitch Shift Semitones",
                    info="Set the pitch shift semitones.",
                    value=0,
                    interactive=True,
                    visible=False,
                )
                limiter_batch = gr.Checkbox(
                    label="Limiter",
                    info="Apply limiter to the audio.",
                    value=False,
                    interactive=True,
                    visible=False,
                )
                limiter_threshold_batch = gr.Slider(
                    minimum=-60,
                    maximum=0,
                    label="Limiter Threshold dB",
                    info="Set the limiter threshold dB.",
                    value=-6,
                    interactive=True,
                    visible=False,
                )

                limiter_release_time_batch = gr.Slider(
                    minimum=0.01,
                    maximum=1,
                    label="Limiter Release Time",
                    info="Set the limiter release time.",
                    value=0.05,
                    interactive=True,
                    visible=False,
                )
                gain_batch = gr.Checkbox(
                    label="Gain",
                    info="Apply gain to the audio.",
                    value=False,
                    interactive=True,
                    visible=False,
                )
                gain_db_batch = gr.Slider(
                    minimum=-60,
                    maximum=60,
                    label="Gain dB",
                    info="Set the gain dB.",
                    value=0,
                    interactive=True,
                    visible=False,
                )
                distortion_batch = gr.Checkbox(
                    label="Distortion",
                    info="Apply distortion to the audio.",
                    value=False,
                    interactive=True,
                    visible=False,
                )
                distortion_gain_batch = gr.Slider(
                    minimum=-60,
                    maximum=60,
                    label="Distortion Gain",
                    info="Set the distortion gain.",
                    value=25,
                    interactive=True,
                    visible=False,
                )
                chorus_batch = gr.Checkbox(
                    label="chorus",
                    info="Apply chorus to the audio.",
                    value=False,
                    interactive=True,
                    visible=False,
                )
                chorus_rate_batch = gr.Slider(
                    minimum=0,
                    maximum=100,
                    label="Chorus Rate Hz",
                    info="Set the chorus rate Hz.",
                    value=1.0,
                    interactive=True,
                    visible=False,
                )

                chorus_depth_batch = gr.Slider(
                    minimum=0,
                    maximum=1,
                    label="chorus Depth",
                    info="Set the chorus depth.",
                    value=0.25,
                    interactive=True,
                    visible=False,
                )

                chorus_center_delay_batch = gr.Slider(
                    minimum=7,
                    maximum=8,
                    label="chorus Center Delay ms",
                    info="Set the chorus center delay ms.",
                    value=7,
                    interactive=True,
                    visible=False,
                )

                chorus_feedback_batch = gr.Slider(
                    minimum=0,
                    maximum=1,
                    label="chorus Feedback",
                    info="Set the chorus feedback.",
                    value=0.0,
                    interactive=True,
                    visible=False,
                )

                chorus_mix_batch = gr.Slider(
                    minimum=0,
                    maximum=1,
                    label="Chorus Mix",
                    info="Set the chorus mix.",
                    value=0.5,
                    interactive=True,
                    visible=False,
                )
                bitcrush_batch = gr.Checkbox(
                    label="Bitcrush",
                    info="Apply bitcrush to the audio.",
                    value=False,
                    interactive=True,
                    visible=False,
                )
                bitcrush_bit_depth_batch = gr.Slider(
                    minimum=1,
                    maximum=32,
                    label="Bitcrush Bit Depth",
                    info="Set the bitcrush bit depth.",
                    value=8,
                    interactive=True,
                    visible=False,
                )
                clipping_batch = gr.Checkbox(
                    label="Clipping",
                    info="Apply clipping to the audio.",
                    value=False,
                    interactive=True,
                    visible=False,
                )
                clipping_threshold_batch = gr.Slider(
                    minimum=-60,
                    maximum=0,
                    label="Clipping Threshold",
                    info="Set the clipping threshold.",
                    value=-6,
                    interactive=True,
                    visible=False,
                )
                compressor_batch = gr.Checkbox(
                    label="Compressor",
                    info="Apply compressor to the audio.",
                    value=False,
                    interactive=True,
                    visible=False,
                )
                compressor_threshold_batch = gr.Slider(
                    minimum=-60,
                    maximum=0,
                    label="Compressor Threshold dB",
                    info="Set the compressor threshold dB.",
                    value=0,
                    interactive=True,
                    visible=False,
                )

                compressor_ratio_batch = gr.Slider(
                    minimum=1,
                    maximum=20,
                    label="Compressor Ratio",
                    info="Set the compressor ratio.",
                    value=1,
                    interactive=True,
                    visible=False,
                )

                compressor_attack_batch = gr.Slider(
                    minimum=0.0,
                    maximum=100,
                    label="Compressor Attack ms",
                    info="Set the compressor attack ms.",
                    value=1.0,
                    interactive=True,
                    visible=False,
                )

                compressor_release_batch = gr.Slider(
                    minimum=0.01,
                    maximum=100,
                    label="Compressor Release ms",
                    info="Set the compressor release ms.",
                    value=100,
                    interactive=True,
                    visible=False,
                )
                delay_batch = gr.Checkbox(
                    label="Delay",
                    info="Apply delay to the audio.",
                    value=False,
                    interactive=True,
                    visible=False,
                )
                delay_seconds_batch = gr.Slider(
                    minimum=0.0,
                    maximum=5.0,
                    label="Delay Seconds",
                    info="Set the delay seconds.",
                    value=0.5,
                    interactive=True,
                    visible=False,
                )

                delay_feedback_batch = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    label="Delay Feedback",
                    info="Set the delay feedback.",
                    value=0.0,
                    interactive=True,
                    visible=False,
                )

                delay_mix_batch = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    label="Delay Mix",
                    info="Set the delay mix.",
                    value=0.5,
                    interactive=True,
                    visible=False,
                )
                with gr.Accordion("Preset Settings", open=False):
                    with gr.Row():
                        preset_dropdown = gr.Dropdown(
                            label="Select Custom Preset",
                            interactive=True,
                        )
                        presets_batch_refresh_button = gr.Button("Refresh Presets")
                    import_file = gr.File(
                        label="Select file to import",
                        file_count="single",
                        type="filepath",
                        interactive=True,
                    )
                    import_file.change(
                        import_presets_button,
                        inputs=import_file,
                        outputs=[preset_dropdown],
                    )
                    presets_batch_refresh_button.click(
                        refresh_presets, outputs=preset_dropdown
                    )
                    with gr.Row():
                        preset_name_input = gr.Textbox(
                            label="Preset Name",
                            placeholder="Enter preset name",
                        )
                        export_button = gr.Button("Export Preset")
                pitch_batch = gr.Slider(
                    minimum=-24,
                    maximum=24,
                    step=1,
                    label="Pitch",
                    info="Set the pitch of the audio, the higher the value, the higher the pitch.",
                    value=0,
                    interactive=True,
                )
                filter_radius_batch = gr.Slider(
                    minimum=0,
                    maximum=7,
                    label="Filter Radius",
                    info="If the number is greater than or equal to three, employing median filtering on the collected tone results has the potential to decrease respiration.",
                    value=3,
                    step=1,
                    interactive=False,
                    visible=False,
                )
                index_rate_batch = gr.Slider(
                    minimum=0,
                    maximum=1,
                    label="Search Feature Ratio",
                    info="Influence exerted by the index file; a higher value corresponds to greater influence. However, opting for lower values can help mitigate artifacts present in the audio.",
                    value=0.5,
                    interactive=True,
                )
                rms_mix_rate_batch = gr.Slider(
                    minimum=0,
                    maximum=1,
                    label="Volume Envelope",
                    info="Substitute or blend with the volume envelope of the output. The closer the ratio is to 1, the more the output envelope is employed.",
                    value=1,
                    interactive=True,
                )
                protect_batch = gr.Slider(
                    minimum=0,
                    maximum=0.5,
                    label="Protect Voiceless Consonants",
                    info="Safeguard distinct consonants and breathing sounds to prevent electro-acoustic tearing and other artifacts. Pulling the parameter to its maximum value of 0.5 offers comprehensive protection. However, reducing this value might decrease the extent of protection while potentially mitigating the indexing effect.",
                    value=0.3,
                    interactive=True,
                )
                preset_dropdown.change(
                    update_sliders,
                    inputs=preset_dropdown,
                    outputs=[
                        pitch_batch,
                        filter_radius_batch,
                        index_rate_batch,
                        rms_mix_rate_batch,
                        protect_batch,
                    ],
                )
                export_button.click(
                    export_presets_button,
                    inputs=[
                        preset_name_input,
                        pitch,
                        filter_radius,
                        index_rate,
                        rms_mix_rate,
                        protect,
                    ],
                    outputs=[],
                )
                hop_length_batch = gr.Slider(
                    minimum=1,
                    maximum=512,
                    step=1,
                    label="Hop Length",
                    info="Denotes the duration it takes for the system to transition to a significant pitch change. Smaller hop lengths require more time for inference but tend to yield higher pitch accuracy.",
                    visible=False,
                    value=160,
                    interactive=True,
                )
                f0_method_batch = gr.Radio(
                    label="Pitch extraction algorithm",
                    info="Pitch extraction algorithm to use for the audio conversion. The default algorithm is rmvpe, which is ***recommended for most cases.***",
                    choices=[
                        "crepe",
                        "crepe-tiny",
                        "rmvpe",
                        "fcpe",
                    ],
                    value="rmvpe",
                    interactive=True,
                )
                embedder_model_batch = gr.Radio(
                    label="Embedder Model",
                    info="Model used for learning speaker embedding.",
                    choices=[
                        "contentvec",
                        "spin",
                        "chinese-hubert-base",
                        "japanese-hubert-base",
                        "korean-hubert-base",
                        "custom",
                    ],
                    value="contentvec",
                    interactive=True,
                )
                f0_file_batch = gr.File(
                    label="The f0 curve represents the variations in the base frequency of a voice over time, showing how pitch rises and falls.",
                    visible=True,
                )
                with gr.Column(visible=False) as embedder_custom_batch:
                    with gr.Accordion("Custom Embedder", open=True):
                        with gr.Row():
                            embedder_model_custom_batch = gr.Dropdown(
                                label="Select Custom Embedder",
                                choices=refresh_embedders_folders(),
                                interactive=True,
                                allow_custom_value=True,
                            )
                            refresh_embedders_button_batch = gr.Button("Refresh embedders")
                        folder_name_input_batch = gr.Textbox(
                            label="Folder Name", interactive=True
                        )
                        with gr.Row():
                            bin_file_upload_batch = gr.File(
                                label="Upload .bin",
                                type="filepath",
                                interactive=True,
                            )
                            config_file_upload_batch = gr.File(
                                label="Upload .json",
                                type="filepath",
                                interactive=True,
                            )
                        move_files_button_batch = gr.Button("Move files to custom embedder folder")

        terms_checkbox_batch = gr.Checkbox(
            label="I agree to the terms of use",
            info="Please ensure compliance with the terms and conditions detailed in [this document](https://github.com/IAHispano/Applio/blob/main/TERMS_OF_USE.md) before proceeding with your inference.",
            value=False,
            interactive=True,
        )
        convert_button_batch = gr.Button("Convert")
        stop_button = gr.Button("Stop convert", visible=False)
        stop_button.click(fn=stop_infer, inputs=[], outputs=[])

        with gr.Row():
            vc_output3 = gr.Textbox(
                label="Output Information",
                info="The output information will be displayed here.",
            )

    def toggle_visible(checkbox):
        return {"visible": checkbox, "__type__": "update"}

    def toggle_visible_hop_length(f0_method):
        if f0_method == "crepe" or f0_method == "crepe-tiny":
            return {"visible": True, "__type__": "update"}
        return {"visible": False, "__type__": "update"}

    def toggle_visible_embedder_custom(embedder_model):
        if embedder_model == "custom":
            return {"visible": True, "__type__": "update"}
        return {"visible": False, "__type__": "update"}

    def enable_stop_convert_button():
        return {"visible": False, "__type__": "update"}, {
            "visible": True,
            "__type__": "update",
        }

    def disable_stop_convert_button():
        return {"visible": True, "__type__": "update"}, {
            "visible": False,
            "__type__": "update",
        }

    def toggle_visible_formant_shifting(checkbox):
        if checkbox:
            return (
                gr.update(visible=True),
                gr.update(visible=True),
                gr.update(visible=True),
                gr.update(visible=True),
                gr.update(visible=True),
            )
        else:
            return (
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
            )

    def update_visibility(checkbox, count):
        return [gr.update(visible=checkbox) for _ in range(count)]

    def post_process_visible(checkbox):
        return update_visibility(checkbox, 10)

    def reverb_visible(checkbox):
        return update_visibility(checkbox, 6)

    def limiter_visible(checkbox):
        return update_visibility(checkbox, 2)

    def chorus_visible(checkbox):
        return update_visibility(checkbox, 6)

    def bitcrush_visible(checkbox):
        return update_visibility(checkbox, 1)

    def compress_visible(checkbox):
        return update_visibility(checkbox, 4)

    def delay_visible(checkbox):
        return update_visibility(checkbox, 3)

    autotune.change(
        fn=toggle_visible,
        inputs=[autotune],
        outputs=[autotune_strength],
    )
    clean_audio.change(
        fn=toggle_visible,
        inputs=[clean_audio],
        outputs=[clean_strength],
    )
    formant_shifting.change(
        fn=toggle_visible_formant_shifting,
        inputs=[formant_shifting],
        outputs=[
            formant_row,
            formant_preset,
            formant_refresh_button,
            formant_qfrency,
            formant_timbre,
        ],
    )
    formant_shifting_batch.change(
        fn=toggle_visible_formant_shifting,
        inputs=[formant_shifting],
        outputs=[
            formant_row_batch,
            formant_preset_batch,
            formant_refresh_button_batch,
            formant_qfrency_batch,
            formant_timbre_batch,
        ],
    )
    formant_refresh_button.click(
        fn=refresh_formant,
        inputs=[],
        outputs=[formant_preset],
    )
    formant_preset.change(
        fn=update_sliders_formant,
        inputs=[formant_preset],
        outputs=[
            formant_qfrency,
            formant_timbre,
        ],
    )
    formant_preset_batch.change(
        fn=update_sliders_formant,
        inputs=[formant_preset_batch],
        outputs=[
            formant_qfrency,
            formant_timbre,
        ],
    )
    post_process.change(
        fn=post_process_visible,
        inputs=[post_process],
        outputs=[
            reverb,
            pitch_shift,
            limiter,
            gain,
            distortion,
            chorus,
            bitcrush,
            clipping,
            compressor,
            delay,
        ],
    )

    reverb.change(
        fn=reverb_visible,
        inputs=[reverb],
        outputs=[
            reverb_room_size,
            reverb_damping,
            reverb_wet_gain,
            reverb_dry_gain,
            reverb_width,
            reverb_freeze_mode,
        ],
    )
    pitch_shift.change(
        fn=toggle_visible,
        inputs=[pitch_shift],
        outputs=[pitch_shift_semitones],
    )
    limiter.change(
        fn=limiter_visible,
        inputs=[limiter],
        outputs=[limiter_threshold, limiter_release_time],
    )
    gain.change(
        fn=toggle_visible,
        inputs=[gain],
        outputs=[gain_db],
    )
    distortion.change(
        fn=toggle_visible,
        inputs=[distortion],
        outputs=[distortion_gain],
    )
    chorus.change(
        fn=chorus_visible,
        inputs=[chorus],
        outputs=[
            chorus_rate,
            chorus_depth,
            chorus_center_delay,
            chorus_feedback,
            chorus_mix,
        ],
    )
    bitcrush.change(
        fn=bitcrush_visible,
        inputs=[bitcrush],
        outputs=[bitcrush_bit_depth],
    )
    clipping.change(
        fn=toggle_visible,
        inputs=[clipping],
        outputs=[clipping_threshold],
    )
    compressor.change(
        fn=compress_visible,
        inputs=[compressor],
        outputs=[
            compressor_threshold,
            compressor_ratio,
            compressor_attack,
            compressor_release,
        ],
    )
    delay.change(
        fn=delay_visible,
        inputs=[delay],
        outputs=[delay_seconds, delay_feedback, delay_mix],
    )
    post_process_batch.change(
        fn=post_process_visible,
        inputs=[post_process_batch],
        outputs=[
            reverb_batch,
            pitch_shift_batch,
            limiter_batch,
            gain_batch,
            distortion_batch,
            chorus_batch,
            bitcrush_batch,
            clipping_batch,
            compressor_batch,
            delay_batch,
        ],
    )

    reverb_batch.change(
        fn=reverb_visible,
        inputs=[reverb_batch],
        outputs=[
            reverb_room_size_batch,
            reverb_damping_batch,
            reverb_wet_gain_batch,
            reverb_dry_gain_batch,
            reverb_width_batch,
            reverb_freeze_mode_batch,
        ],
    )
    pitch_shift_batch.change(
        fn=toggle_visible,
        inputs=[pitch_shift_batch],
        outputs=[pitch_shift_semitones_batch],
    )
    limiter_batch.change(
        fn=limiter_visible,
        inputs=[limiter_batch],
        outputs=[limiter_threshold_batch, limiter_release_time_batch],
    )
    gain_batch.change(
        fn=toggle_visible,
        inputs=[gain_batch],
        outputs=[gain_db_batch],
    )
    distortion_batch.change(
        fn=toggle_visible,
        inputs=[distortion_batch],
        outputs=[distortion_gain_batch],
    )
    chorus_batch.change(
        fn=chorus_visible,
        inputs=[chorus_batch],
        outputs=[
            chorus_rate_batch,
            chorus_depth_batch,
            chorus_center_delay_batch,
            chorus_feedback_batch,
            chorus_mix_batch,
        ],
    )
    bitcrush_batch.change(
        fn=bitcrush_visible,
        inputs=[bitcrush_batch],
        outputs=[bitcrush_bit_depth_batch],
    )
    clipping_batch.change(
        fn=toggle_visible,
        inputs=[clipping_batch],
        outputs=[clipping_threshold_batch],
    )
    compressor_batch.change(
        fn=compress_visible,
        inputs=[compressor_batch],
        outputs=[
            compressor_threshold_batch,
            compressor_ratio_batch,
            compressor_attack_batch,
            compressor_release_batch,
        ],
    )
    delay_batch.change(
        fn=delay_visible,
        inputs=[delay_batch],
        outputs=[delay_seconds_batch, delay_feedback_batch, delay_mix_batch],
    )
    autotune_batch.change(
        fn=toggle_visible,
        inputs=[autotune_batch],
        outputs=[autotune_strength_batch],
    )
    clean_audio_batch.change(
        fn=toggle_visible,
        inputs=[clean_audio_batch],
        outputs=[clean_strength_batch],
    )
    f0_method.change(
        fn=toggle_visible_hop_length,
        inputs=[f0_method],
        outputs=[hop_length],
    )
    f0_method_batch.change(
        fn=toggle_visible_hop_length,
        inputs=[f0_method_batch],
        outputs=[hop_length_batch],
    )
    refresh_button.click(
        fn=change_choices,
        inputs=[model_file],
        outputs=[model_file, index_file, audio_dropdown, sid, sid_batch],
    )
    audio.change(
        fn=output_path_fn,
        inputs=[audio],
        outputs=[output_path],
    )
    upload_audio.upload(
        fn=save_to_wav2,
        inputs=[upload_audio],
        outputs=[audio, output_path],
    )
    upload_audio.stop_recording(
        fn=save_to_wav,
        inputs=[upload_audio],
        outputs=[audio, output_path],
    )
    
    audio_dropdown.change(
        fn=lambda selected: (selected, output_path_fn(selected) if selected else ""),
        inputs=[audio_dropdown],
        outputs=[audio, output_path],
    )
    
    audio.change(
        fn=lambda youtube_url: output_path_fn(youtube_url) if youtube_url and youtube_url.startswith("http") else "",
        inputs=[audio],
        outputs=[output_path],
    )

    clear_outputs_infer.click(
        fn=delete_outputs,
        inputs=[],
        outputs=[],
    )
    clear_outputs_batch.click(
        fn=delete_outputs,
        inputs=[],
        outputs=[],
    )
    embedder_model.change(
        fn=toggle_visible_embedder_custom,
        inputs=[embedder_model],
        outputs=[embedder_custom],
    )
    embedder_model_batch.change(
        fn=toggle_visible_embedder_custom,
        inputs=[embedder_model_batch],
        outputs=[embedder_custom_batch],
    )
    move_files_button.click(
        fn=create_folder_and_move_files,
        inputs=[folder_name_input, bin_file_upload, config_file_upload],
        outputs=[],
    )
    refresh_embedders_button.click(
        fn=lambda: gr.update(choices=refresh_embedders_folders()),
        inputs=[],
        outputs=[embedder_model_custom],
    )
    move_files_button_batch.click(
        fn=create_folder_and_move_files,
        inputs=[
            folder_name_input_batch,
            bin_file_upload_batch,
            config_file_upload_batch,
        ],
        outputs=[],
    )
    refresh_embedders_button_batch.click(
        fn=lambda: gr.update(choices=refresh_embedders_folders()),
        inputs=[],
        outputs=[embedder_model_custom_batch],
    )
    convert_button1.click(
        fn=enforce_terms,
        inputs=[
            terms_checkbox,
            pitch,
            filter_radius,
            index_rate,
            rms_mix_rate,
            protect,
            hop_length,
            f0_method,
            audio,
            has_bg_music,
            output_path,
            model_file,
            index_file,
            split_audio,
            autotune,
            autotune_strength,
            clean_audio,
            clean_strength,
            export_format,
            f0_file,
            embedder_model,
            embedder_model_custom,
            formant_shifting,
            formant_qfrency,
            formant_timbre,
            post_process,
            reverb,
            pitch_shift,
            limiter,
            gain,
            distortion,
            chorus,
            bitcrush,
            clipping,
            compressor,
            delay,
            reverb_room_size,
            reverb_damping,
            reverb_wet_gain,
            reverb_dry_gain,
            reverb_width,
            reverb_freeze_mode,
            pitch_shift_semitones,
            limiter_threshold,
            limiter_release_time,
            gain_db,
            distortion_gain,
            chorus_rate,
            chorus_depth,
            chorus_center_delay,
            chorus_feedback,
            chorus_mix,
            bitcrush_bit_depth,
            clipping_threshold,
            compressor_threshold,
            compressor_ratio,
            compressor_attack,
            compressor_release,
            delay_seconds,
            delay_feedback,
            delay_mix,
            sid,
        ],
        outputs=[vc_output1, vc_output2],
    )
    convert_button_batch.click(
        fn=enforce_terms_batch,
        inputs=[
            terms_checkbox_batch,
            pitch_batch,
            filter_radius_batch,
            index_rate_batch,
            rms_mix_rate_batch,
            protect_batch,
            hop_length_batch,
            f0_method_batch,
            input_folder_batch,
            output_folder_batch,
            model_file,
            index_file,
            split_audio_batch,
            autotune_batch,
            autotune_strength_batch,
            clean_audio_batch,
            clean_strength_batch,
            export_format_batch,
            f0_file_batch,
            embedder_model_batch,
            embedder_model_custom_batch,
            formant_shifting_batch,
            formant_qfrency_batch,
            formant_timbre_batch,
            post_process_batch,
            reverb_batch,
            pitch_shift_batch,
            limiter_batch,
            gain_batch,
            distortion_batch,
            chorus_batch,
            bitcrush_batch,
            clipping_batch,
            compressor_batch,
            delay_batch,
            reverb_room_size_batch,
            reverb_damping_batch,
            reverb_wet_gain_batch,
            reverb_dry_gain_batch,
            reverb_width_batch,
            reverb_freeze_mode_batch,
            pitch_shift_semitones_batch,
            limiter_threshold_batch,
            limiter_release_time_batch,
            gain_db_batch,
            distortion_gain_batch,
            chorus_rate_batch,
            chorus_depth_batch,
            chorus_center_delay_batch,
            chorus_feedback_batch,
            chorus_mix_batch,
            bitcrush_bit_depth_batch,
            clipping_threshold_batch,
            compressor_threshold_batch,
            compressor_ratio_batch,
            compressor_attack_batch,
            compressor_release_batch,
            delay_seconds_batch,
            delay_feedback_batch,
            delay_mix_batch,
            sid_batch,
        ],
        outputs=[vc_output3],
    )
    convert_button_batch.click(
        fn=enable_stop_convert_button,
        inputs=[],
        outputs=[convert_button_batch, stop_button],
    )
    stop_button.click(
        fn=disable_stop_convert_button,
        inputs=[],
        outputs=[convert_button_batch, stop_button],
    )

def get_hash(filepath):
    """Generate hash for local audio files"""
    with open(filepath, 'rb') as f:
        file_hash = hashlib.blake2b()
        while chunk := f.read(8192):
            file_hash.update(chunk)
    return file_hash.hexdigest()[:11]


def get_youtube_video_id(url, ignore_playlist=True):
    """Extract YouTube video ID from URL"""
    query = urlparse(url)
    if query.hostname == 'youtu.be':
        if query.path[1:] == 'watch':
            return query.query[2:]
        return query.path[1:]

    if query.hostname in {'www.youtube.com', 'youtube.com', 'music.youtube.com'}:
        if not ignore_playlist:
            with suppress(KeyError):
                return parse_qs(query.query)['list'][0]
        if query.path == '/watch':
            return parse_qs(query.query)['v'][0]
        if query.path[:7] == '/watch/':
            return query.path.split('/')[1]
        if query.path[:7] == '/embed/':
            return query.path.split('/')[2]
        if query.path[:3] == '/v/':
            return query.path.split('/')[2]
    return None