# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
This example shows how to use vLLM-Omni for running offline inference
with the correct prompt format on Qwen2.5-Omni
"""

import os
from typing import NamedTuple, Optional

import librosa
import numpy as np
import soundfile as sf
from PIL import Image
from vllm.assets.audio import AudioAsset
from vllm.assets.image import ImageAsset
from vllm.assets.video import VideoAsset, video_to_ndarrays
from vllm.multimodal.image import convert_image_mode
from vllm.sampling_params import SamplingParams
from vllm.utils.argparse_utils import FlexibleArgumentParser


from vllm_omni.entrypoints.omni import Omni

SEED = 42


class QueryResult(NamedTuple):
    inputs: dict
    limit_mm_per_prompt: dict[str, int]


# NOTE: The default `max_num_seqs` and `max_model_len` may result in OOM on
# lower-end GPUs.
# Unless specified, these settings have been tested to work on a single L4.

default_system = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba "
    "Group, capable of perceiving auditory and visual inputs, as well as "
    "generating text and speech."
)


def get_text_query(question: str = None) -> QueryResult:
    if question is None:
        question = "Explain the system architecture for a scalable audio generation pipeline. Answer in 15 words."
    prompt = (
        f"<|im_start|>system\n{default_system}<|im_end|>\n"
        "<|im_start|>user\n"
        f"{question}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    return QueryResult(
        inputs={
            "prompt": prompt,
        },
        limit_mm_per_prompt={},
    )


def get_mixed_modalities_query(
    video_path: Optional[str] = None,
    image_path: Optional[str] = None,
    audio_path: Optional[str] = None,
    num_frames: int = 16,
    sampling_rate: int = 16000,
) -> QueryResult:
    question = "What is recited in the audio? What is the content of this image? Why is this video funny?"
    prompt = (
        f"<|im_start|>system\n{default_system}<|im_end|>\n"
        "<|im_start|>user\n<|audio_bos|><|AUDIO|><|audio_eos|>"
        "<|vision_bos|><|IMAGE|><|vision_eos|>"
        "<|vision_bos|><|VIDEO|><|vision_eos|>"
        f"{question}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    # Load video
    if video_path:
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        video_frames = video_to_ndarrays(video_path, num_frames=num_frames)
    else:
        video_frames = VideoAsset(name="baby_reading", num_frames=num_frames).np_ndarrays

    # Load image
    if image_path:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        pil_image = Image.open(image_path)
        image_data = convert_image_mode(pil_image, "RGB")
    else:
        image_data = convert_image_mode(ImageAsset("cherry_blossom").pil_image, "RGB")

    # Load audio
    if audio_path:
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        audio_signal, sr = librosa.load(audio_path, sr=sampling_rate)
        audio_data = (audio_signal.astype(np.float32), sr)
    else:
        audio_data = AudioAsset("mary_had_lamb").audio_and_sample_rate

    return QueryResult(
        inputs={
            "prompt": prompt,
            "multi_modal_data": {
                "audio": audio_data,
                "image": image_data,
                "video": video_frames,
            },
        },
        limit_mm_per_prompt={"audio": 1, "image": 1, "video": 1},
    )


def get_use_audio_in_video_query(
    video_path: Optional[str] = None, num_frames: int = 16, sampling_rate: int = 16000
) -> QueryResult:
    question = "Describe the content of the video, then convert what the baby say into text."
    prompt = (
        f"<|im_start|>system\n{default_system}<|im_end|>\n"
        "<|im_start|>user\n<|vision_bos|><|VIDEO|><|vision_eos|>"
        f"{question}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    if video_path:
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        video_frames = video_to_ndarrays(video_path, num_frames=num_frames)
        # Extract audio from video file
        audio_signal, sr = librosa.load(video_path, sr=sampling_rate)
        audio = (audio_signal.astype(np.float32), sr)
    else:
        asset = VideoAsset(name="baby_reading", num_frames=num_frames)
        video_frames = asset.np_ndarrays
        audio = asset.get_audio(sampling_rate=sampling_rate)

    return QueryResult(
        inputs={
            "prompt": prompt,
            "multi_modal_data": {
                "video": video_frames,
                "audio": audio,
            },
            "mm_processor_kwargs": {
                "use_audio_in_video": True,
            },
        },
        limit_mm_per_prompt={"audio": 1, "video": 1},
    )


def get_multi_audios_query(audio_path: Optional[str] = None, sampling_rate: int = 16000) -> QueryResult:
    question = "Are these two audio clips the same?"
    prompt = (
        f"<|im_start|>system\n{default_system}<|im_end|>\n"
        "<|im_start|>user\n<|audio_bos|><|AUDIO|><|audio_eos|>"
        "<|audio_bos|><|AUDIO|><|audio_eos|>"
        f"{question}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    if audio_path:
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        audio_signal, sr = librosa.load(audio_path, sr=sampling_rate)
        audio_data = (audio_signal.astype(np.float32), sr)
        # Use the provided audio as the first audio, default as second
        audio_list = [
            audio_data,
            AudioAsset("mary_had_lamb").audio_and_sample_rate,
        ]
    else:
        audio_list = [
            AudioAsset("winning_call").audio_and_sample_rate,
            AudioAsset("mary_had_lamb").audio_and_sample_rate,
        ]

    return QueryResult(
        inputs={
            "prompt": prompt,
            "multi_modal_data": {
                "audio": audio_list,
            },
        },
        limit_mm_per_prompt={
            "audio": 2,
        },
    )


def get_image_query(question: str = None, image_path: Optional[str] = None) -> QueryResult:
    if question is None:
        question = "What is the content of this image?"
    prompt = (
        f"<|im_start|>system\n{default_system}<|im_end|>\n"
        "<|im_start|>user\n<|vision_bos|><|IMAGE|><|vision_eos|>"
        f"{question}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    if image_path:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        pil_image = Image.open(image_path)
        image_data = convert_image_mode(pil_image, "RGB")
    else:
        image_data = convert_image_mode(ImageAsset("cherry_blossom").pil_image, "RGB")

    return QueryResult(
        inputs={
            "prompt": prompt,
            "multi_modal_data": {
                "image": image_data,
            },
        },
        limit_mm_per_prompt={"image": 1},
    )


def get_video_query(question: str = None, video_path: Optional[str] = None, num_frames: int = 16) -> QueryResult:
    if question is None:
        question = "Why is this video funny?"
    prompt = (
        f"<|im_start|>system\n{default_system}<|im_end|>\n"
        "<|im_start|>user\n<|vision_bos|><|VIDEO|><|vision_eos|>"
        f"{question}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    if video_path:
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        video_frames = video_to_ndarrays(video_path, num_frames=num_frames)
    else:
        video_frames = VideoAsset(name="baby_reading", num_frames=num_frames).np_ndarrays

    return QueryResult(
        inputs={
            "prompt": prompt,
            "multi_modal_data": {
                "video": video_frames,
            },
        },
        limit_mm_per_prompt={"video": 1},
    )


def get_audio_query(question: str = None, audio_path: Optional[str] = None, sampling_rate: int = 16000) -> QueryResult:
    if question is None:
        question = "What is the content of this audio?"
    prompt = (
        f"<|im_start|>system\n{default_system}<|im_end|>\n"
        "<|im_start|>user\n<|audio_bos|><|AUDIO|><|audio_eos|>"
        f"{question}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    if audio_path:
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        audio_signal, sr = librosa.load(audio_path, sr=sampling_rate)
        audio_data = (audio_signal.astype(np.float32), sr)
    else:
        audio_data = AudioAsset("mary_had_lamb").audio_and_sample_rate

    return QueryResult(
        inputs={
            "prompt": prompt,
            "multi_modal_data": {
                "audio": audio_data,
            },
        },
        limit_mm_per_prompt={"audio": 1},
    )


query_map = {
    "mixed_modalities": get_mixed_modalities_query,
    "use_audio_in_video": get_use_audio_in_video_query,
    "multi_audios": get_multi_audios_query,
    "use_image": get_image_query,
    "use_video": get_video_query,
    "use_audio": get_audio_query,
    "text": get_text_query,
}


def main(args):
    model_name = "Qwen/Qwen2.5-Omni-7B"

    # Get paths from args
    video_path = getattr(args, "video_path", None)
    image_path = getattr(args, "image_path", None)
    audio_path = getattr(args, "audio_path", None)
    num_frames = getattr(args, "num_frames", 16)
    sampling_rate = getattr(args, "sampling_rate", 16000)

    # Get the query function and call it with appropriate parameters
    query_func = query_map[args.query_type]
    if args.query_type == "mixed_modalities":
        query_result = query_func(
            video_path=video_path,
            image_path=image_path,
            audio_path=audio_path,
            num_frames=num_frames,
            sampling_rate=sampling_rate,
        )
    elif args.query_type == "use_audio_in_video":
        query_result = query_func(video_path=video_path, num_frames=num_frames, sampling_rate=sampling_rate)
    elif args.query_type == "multi_audios":
        query_result = query_func(audio_path=audio_path, sampling_rate=sampling_rate)
    elif args.query_type == "use_image":
        query_result = query_func(image_path=image_path)
    elif args.query_type == "use_video":
        query_result = query_func(video_path=video_path, num_frames=num_frames)
    elif args.query_type == "use_audio":
        query_result = query_func(audio_path=audio_path, sampling_rate=sampling_rate)
    else:
        query_result = query_func()

    omni_llm = Omni(
        model=model_name,
        log_stats=args.enable_stats,
        log_file=("omni_llm_pipeline.log" if args.enable_stats else None),
        init_sleep_seconds=args.init_sleep_seconds,
        batch_timeout=args.batch_timeout,
        init_timeout=args.init_timeout,
        shm_threshold_bytes=args.shm_threshold_bytes,
    )
    thinker_sampling_params = SamplingParams(
        temperature=0.0,  # Deterministic - no randomness
        top_p=1.0,  # Disable nucleus sampling
        top_k=-1,  # Disable top-k sampling
        max_tokens=2048,
        seed=SEED,  # Fixed seed for sampling
        detokenize=True,
        repetition_penalty=1.1,
    )
    talker_sampling_params = SamplingParams(
        temperature=0.9,
        top_p=0.8,
        top_k=40,
        max_tokens=2048,
        seed=SEED,  # Fixed seed for sampling
        detokenize=True,
        repetition_penalty=1.05,
        stop_token_ids=[8294],
    )
    code2wav_sampling_params = SamplingParams(
        temperature=0.0,  # Deterministic - no randomness
        top_p=1.0,  # Disable nucleus sampling
        top_k=-1,  # Disable top-k sampling
        max_tokens=2048,
        seed=SEED,  # Fixed seed for sampling
        detokenize=True,
        repetition_penalty=1.1,
    )

    sampling_params_list = [
        thinker_sampling_params,
        talker_sampling_params,
        code2wav_sampling_params,
    ]

    if args.txt_prompts is None:
        prompts = [query_result.inputs for _ in range(args.num_prompts)]
    else:
        assert args.query_type == "text", "txt-prompts is only supported for text query type"
        with open(args.txt_prompts, encoding="utf-8") as f:
            lines = [ln.strip() for ln in f.readlines()]
            prompts = [get_text_query(ln).inputs for ln in lines if ln != ""]
            print(f"[Info] Loaded {len(prompts)} prompts from {args.txt_prompts}")
    # from vllm_omni.inputs.data import OmniTokensPrompt
    # prompts = OmniTokensPrompt(prompt_token_ids=[3380, 1835, 6379, 2557, 2557, 2557, 2557, 2557, 6379, 2132, 6299, 6591, 3550, 7802, 8014, 7320, 6835, 1587, 4543, 4543, 4543, 578, 4536, 3344, 7015, 629, 629, 6788, 1220, 1220, 7075, 3610, 3783, 3783, 1267, 6567, 6567, 7607, 7607, 4055, 3064, 1722, 7089, 4632, 8120, 7793, 10, 5936, 1846, 5384, 7190, 5605, 2313, 444, 1672, 6470, 6270, 6391, 8117, 7945, 5977, 1301, 556, 1856, 137, 2226, 5683, 1499, 6426, 3327, 6665, 5182, 3303, 1158, 1600, 3493, 1248, 3999, 3757, 3904, 4340, 3640, 3578, 4787, 7595, 3534, 370, 1134, 1686, 3083, 4575, 5455, 3333, 7066, 4770, 1339, 8137, 5087, 3355, 2703, 3138, 6821, 4440, 51, 7485, 850, 2534, 2207, 87, 1452, 6765, 3937, 321, 1772, 4647, 4751, 4751, 2320, 2578, 3075, 6350, 2045, 3352, 4852, 4690, 6648, 2400, 2864, 7940, 87, 2118, 843, 5278, 2874, 3998, 7861, 1434, 7324, 3442, 1914, 1480, 672, 1021, 27, 3431, 6252, 678, 678, 3689, 3689, 5852, 5718, 7965, 2840, 5546, 179, 939, 665, 6097, 1843, 1026, 5854, 1652, 7045, 966, 1652, 966, 279, 279, 1652, 1026, 1026, 2113, 1081, 237, 4127, 287, 2830, 885, 2398, 6904, 5085, 4456, 201, 894, 6678, 418, 1708, 7988, 351, 1665, 7728, 1365, 4380, 733, 3439, 7728, 4300, 3640, 5161, 3380, 7395, 7431, 5017, 1819, 4599, 1898, 4559, 7145, 7108, 7130, 2040, 876, 8007, 6260, 1905, 1737, 2296, 4138, 5039, 2588, 1205, 1899, 4513, 5649, 2779, 1089, 3981, 6383, 4708, 1949, 5634, 538, 1548, 1205, 5701, 3432, 6739, 5064, 2355, 1327, 1667, 3874, 2464, 2464, 8182, 2058, 243, 4620, 7450, 7248, 4564, 5099, 3410, 3757, 2062, 2062, 6408, 6568, 2918, 4986, 516, 516, 110, 936, 7847, 1597, 1552, 1552, 1552, 1552, 1224, 1224, 1504, 7497, 2118, 1318, 1179, 2439, 429, 2662, 3855, 7440, 5690, 1062, 3577, 798, 1552, 1746, 2446, 7810, 3765, 2812, 1870, 6253, 1019, 4684, 5873, 3297, 6364, 2643, 2335, 7064, 2421, 1789, 723, 4494, 6735, 534, 1012, 7332, 4077, 5852, 3676, 5629, 2010, 204, 7432, 7432, 5099, 5869, 6645, 5808, 5808, 6340, 3640, 6943, 5113, 3498, 7497, 6182, 5927, 1026, 1026, 279, 279, 279, 279, 887, 2688, 5927, 2270, 5546, 2362, 179, 4850, 924, 5884, 3028, 5668, 616, 524, 7202, 6329, 778, 1637, 6379, 7658, 3113, 5455, 5283, 310, 660, 4694, 5648, 4057, 7036, 3069, 6857, 4276, 5824, 1284, 6315, 1503, 7998, 536, 5201, 8006, 822, 578, 8164, 5059, 972, 3123, 4201, 6125, 2406, 6125, 5319, 1434, 274, 5241, 632, 2199, 8087, 2541, 3338, 4694, 613, 4204, 2643, 3597, 6009, 3363, 3134, 1439, 39, 3694, 552, 2726, 376, 613, 5813, 5643, 2604, 2604, 549, 1670, 1424, 349, 3689, 4682, 3788, 3365, 7800, 3843, 3843, 7612, 2608, 1267, 6567, 2830, 665, 1497, 1064, 1064, 1652, 7045, 887, 887, 279, 279, 279, 5447, 8106, 7497, 1491, 8049, 280, 6086, 1044, 6768, 7595, 4112, 7466, 3259, 1238, 1468, 6340, 5412, 5740, 3469, 2089, 2104, 4535, 1434, 1434, 19, 2185, 5683, 4219, 5327, 4769, 2850, 6518, 109, 431, 6302, 4954, 510, 7608, 5019, 5019, 2521, 7590, 2605, 6997, 3404, 1372, 1235, 1852, 5156, 4445, 4499, 7575, 4882, 3302, 7708, 5554, 5341, 2643, 2690, 3727, 846, 288, 2830, 146, 2211, 4410, 7852, 4339, 3303, 3648, 3837, 7857, 5056, 5056, 6548, 2726, 1303, 555, 4219, 1303, 857, 6785, 721, 6014, 2247, 6464, 3651, 3329, 4318, 1519, 946, 7609, 2004, 1197, 7341, 7777, 8129, 668, 5892, 5892, 6874, 1672, 4445, 3252, 5972, 5345, 7268, 2343, 1372, 3427, 3307, 1424, 1424, 8103, 4120, 1047, 4, 7928, 1827, 6980, 2834, 1722, 4101, 2717, 5841, 2782, 5981, 3712, 2489, 4239, 6113, 170, 4680, 2412, 3280, 471, 4464, 3125, 1358, 6670, 1224, 2690, 2722, 4728, 6443, 1306, 2655, 3072, 1852, 3600, 3944, 1744, 632, 4079, 4012, 7969, 938, 6940, 2146, 2743, 2312, 4513, 7728, 5296, 7835, 7789, 2581, 1874, 5649, 3640, 784, 3111, 7192, 7494, 2329, 5390, 6948, 3418, 6512, 3221, 204, 462, 5499, 3882, 716, 7671, 5808, 3640, 3640, 4957, 1973, 1973, 6379, 6379, 4850, 1524, 4489, 6097, 6097, 6755, 6755, 1026, 4191, 1026, 1026, 1026, 1026, 1026, 1026, 4191, 6264, 6264, 6264, 6264, 4489, 4489, 4489, 4489, 4314, 2636, 158, 158, 2472, 469, 4850, 179, 2557, 2557, 2557, 1053, 4263, 6278, 384, 8100, 3766, 1629, 5892, 5565, 3810, 6470, 5972, 6902, 4929, 5375, 7415, 4368, 1752, 3424, 416, 1488, 2479, 3040, 1016, 1405, 4266, 2764, 416, 2906, 3744, 1490, 7549, 276, 4342, 7933, 7933, 1112, 3214, 3214, 4752, 333, 8182, 5056, 7857, 5592, 7448, 5264, 917, 1587, 5634, 5668, 228, 4353, 4756, 1485, 4694, 349, 6523, 4680, 972, 2995, 915, 4559, 2530, 2978, 6630, 2118, 2398, 5283, 348, 5513, 7079, 7964, 6332, 1602, 5056, 2321, 1912, 3425, 7789, 7789, 4524, 970, 4432, 5434, 5413, 428, 74, 3826, 723, 8098, 7428, 2395, 5414, 3640, 3640, 2381, 3526, 3285, 5036, 2719, 699, 7397, 3531, 7502, 3218, 1303, 4166, 299, 3069, 4852, 4852, 4852, 6648, 3345, 4298, 5929, 4891, 7516, 1595, 6859, 6984, 4927, 1402, 4590, 6336, 4667, 8116, 5586, 4267, 255, 2947, 1961, 3047, 4161, 1679, 1679, 5601, 7598, 7598, 7914, 7575, 7575, 3640, 1665, 5649, 2051, 109, 245, 8085, 3797, 6874, 6874, 674, 7129, 2649, 6379, 2983, 2983, 6691, 6691, 1453, 2983, 505, 8294])
    omni_outputs = omni_llm.generate(prompts, sampling_params_list)

    # Determine output directory: prefer --output-dir; fallback to --output-wav
    output_dir = args.output_dir if getattr(args, "output_dir", None) else args.output_wav
    os.makedirs(output_dir, exist_ok=True)
    for stage_outputs in omni_outputs:
        if stage_outputs.final_output_type == "text":
            for output in stage_outputs.request_output:
                request_id = int(output.request_id)
                text_output = output.outputs[0].text
                # Save aligned text file per request
                prompt_text = prompts[request_id]["prompt"]
                out_txt = os.path.join(output_dir, f"{request_id:05d}.txt")
                lines = []
                lines.append("Prompt:\n")
                lines.append(str(prompt_text) + "\n")
                lines.append("vllm_text_output:\n")
                lines.append(str(text_output).strip() + "\n")
                try:
                    with open(out_txt, "w", encoding="utf-8") as f:
                        f.writelines(lines)
                except Exception as e:
                    print(f"[Warn] Failed writing text file {out_txt}: {e}")
                print(f"Request ID: {request_id}, Text saved to {out_txt}")
        elif stage_outputs.final_output_type == "audio":
            for output in stage_outputs.request_output:
                request_id = int(output.request_id)
                audio_tensor = output.multimodal_output["audio"]
                output_wav = os.path.join(output_dir, f"output_{output.request_id}.wav")
                sf.write(output_wav, audio_tensor.detach().cpu().numpy(), samplerate=24000)
                print(f"Request ID: {request_id}, Saved audio to {output_wav}")


def parse_args():
    parser = FlexibleArgumentParser(description="Demo on using vLLM for offline inference with audio language models")
    parser.add_argument(
        "--query-type",
        "-q",
        type=str,
        default="mixed_modalities",
        choices=query_map.keys(),
        help="Query type.",
    )
    parser.add_argument(
        "--enable-stats",
        action="store_true",
        default=False,
        help="Enable writing detailed statistics (default: disabled)",
    )
    parser.add_argument(
        "--init-sleep-seconds",
        type=int,
        default=20,
        help="Sleep seconds after starting each stage process to allow initialization (default: 20)",
    )
    parser.add_argument(
        "--batch-timeout",
        type=int,
        default=5,
        help="Timeout for batching in seconds (default: 5)",
    )
    parser.add_argument(
        "--init-timeout",
        type=int,
        default=300,
        help="Timeout for initializing stages in seconds (default: 300)",
    )
    parser.add_argument(
        "--shm-threshold-bytes",
        type=int,
        default=65536,
        help="Threshold for using shared memory in bytes (default: 65536)",
    )
    parser.add_argument(
        "--output-wav",
        default="output_audio",
        help="[Deprecated] Output wav directory (use --output-dir).",
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=1,
        help="Number of prompts to generate.",
    )
    parser.add_argument(
        "--txt-prompts",
        type=str,
        default=None,
        help="Path to a .txt file with one prompt per line (preferred).",
    )
    parser.add_argument(
        "--video-path",
        "-v",
        type=str,
        default=None,
        help="Path to local video file. If not provided, uses default video asset.",
    )
    parser.add_argument(
        "--image-path",
        "-i",
        type=str,
        default=None,
        help="Path to local image file. If not provided, uses default image asset.",
    )
    parser.add_argument(
        "--audio-path",
        "-a",
        type=str,
        default=None,
        help="Path to local audio file. If not provided, uses default audio asset.",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=16,
        help="Number of frames to extract from video (default: 16).",
    )
    parser.add_argument(
        "--sampling-rate",
        type=int,
        default=16000,
        help="Sampling rate for audio loading (default: 16000).",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
