import os
import random
from pydub import AudioSegment

dataset_path = "./clips_rd/"
take_num = 7
amb_name = "Restaurant ambience"
ambience_path = "./videos/synthesis/ambience/"
output_path = "./videos/synthesis/script_output/"
amb_output_path = "./videos/synthesis/used_ambience/"

min_db = 10
max_db = 20

genSeed = 222

def find_take_parts(food_dir, food_name, take_num):
    prefix = f"{food_name}_{take_num}_"
    parts = [f for f in os.listdir(food_dir)
             if f.startswith(prefix) and f.lower().endswith(".wav")]
    parts.sort()

    return [os.path.join(food_dir, p) for p in parts]

def build_take_audio(parts, needed_ms=60_000):
    audio = AudioSegment.silent(duration=0)
    last = None
    for p in parts:
        seg = AudioSegment.from_wav(p)
        audio += seg
        last = p.split("_")[-1].split(".")[0]

        if len(audio) >= needed_ms:
            break
    return last, audio

def process(food_dirs, output_dir):
    MINUTES = len(food_dirs)
    SEG_LEN_MS = MINUTES * 60 * 1000
    os.makedirs(output_dir, exist_ok=True)

    
    # if len(ambience_files) < len(food_dirs):
    #     raise RuntimeError("Need at least as many ambience files as food takes.")

    ambient = AudioSegment.from_wav(os.path.join(ambience_path, amb_name + ".wav"))

    if len(ambient) < SEG_LEN_MS:
        raise RuntimeError(f"Ambient '{amb_name}' shorter than {MINUTES} min.")

    # Pick random 18-min slice
    start_ms = random.randint(0, len(ambient) - SEG_LEN_MS)
    amb_seg = ambient[start_ms:start_ms + SEG_LEN_MS]

    amb_seg.export(os.path.join(amb_output_path, f"{amb_name}.wav"), format="wav")

    # Split into 60-sec chunks
    amb_chunks = [
        amb_seg[i * 60000:(i + 1) * 60000]
        for i in range(MINUTES)
    ]

    for minute_idx, food_dir in enumerate(food_dirs):
        food_name = food_dir.split("/")[-1]

        parts = find_take_parts(food_dir, food_name, take_num)


        # Build food audio
        last, food_audio = build_take_audio(parts, 60_000)

        if last == None:
            print(f"Warning: Food '{food_dir}', take {take_num} does not exist.")
            continue

        ambient_chunk = amb_chunks[minute_idx]

        if len(food_audio) < 60_000:
            print(f"Warning:  Food '{food_dir}', take {take_num} only {len(food_audio)/1000:.1f}s total.")
            ambient_chunk = ambient_chunk[:len(food_audio)]
        food_slice = food_audio[:60_000]

        # Random attenuation of 20–30 dB 
        attenuation_db = random.uniform(min_db, max_db)
        food_slice = food_slice - attenuation_db

        # Overlay on this minute’s ambient
        mixed = ambient_chunk.overlay(food_slice)

        out_name = f"{amb_name} {food_name}_{last} {take_num} {min_db}-{max_db}"
        out_path = os.path.join(output_dir, out_name+".wav")
        mixed.export(out_path, format="wav")
        print(f"Saved {out_name}")


if __name__ == "__main__":
    random.seed = genSeed

    # No cabbage because of inconsistant loudness
    FOOD_DIRS = [
        "aloe", "burger", "candied_fruits", "carrots",
        "chips", "chocolate", "fries", "grapes", "gummies",
        "ice-cream", "jelly", "noodles", "pickles", "pizza",
        "ribs", "salmon", "wings"
    ]

    # Prepend root to each:
    food_paths = [os.path.join(dataset_path, d) for d in FOOD_DIRS]

    process(
        food_dirs=food_paths,
        output_dir=output_path
    )
