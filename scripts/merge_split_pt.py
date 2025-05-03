import argparse
from pathlib import Path

MODEL_FOLDER_PATH: Path = Path(__file__).resolve().parent.parent / "results" / "03adda57-e689-4a1b-b1aa-84692602e12f"
MODEL_PATH: Path = MODEL_FOLDER_PATH / "model.pt"
CHUNKS_FOLDER_PATH: Path = MODEL_FOLDER_PATH / "chunks"
CHUNKS_FOLDER_PATH.mkdir(parents=False, exist_ok=True)

CHUNK_SIZE_MB: int = 40

def split_pt_file():
    chunk_size = CHUNK_SIZE_MB * 1024 * 1024  # convert MB to bytes

    with open(MODEL_PATH, 'rb') as f:
        i = 0
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            chunk_path: Path = CHUNKS_FOLDER_PATH / f"model.part{i}"
            with open(chunk_path, 'wb') as chunk_file:
                chunk_file.write(chunk)
            i += 1
    print(f"Split into {i} chunks of {CHUNK_SIZE_MB} MB")


def merge_pt_chunks():
    with open(MODEL_PATH, 'wb') as out_file:
        i = 0
        while True:
            part_file: Path = CHUNKS_FOLDER_PATH / f"model.part{i}"
            if not part_file.exists():
                break
            with open(part_file, 'rb') as pf:
                out_file.write(pf.read())
            i += 1
    print(f"Merged chunks into '{MODEL_PATH}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split or merge a .pt file.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    split_parser = subparsers.add_parser("split", help="Split a .pt file into chunks")
    merge_parser = subparsers.add_parser("merge", help="Merge .pt file chunks into one file")
    args = parser.parse_args()

    if args.command == "split":
        split_pt_file()
    elif args.command == "merge":
       merge_pt_chunks()
