"""Fetch the original MTL-KD teachers at a pinned revision and verify SHA-256."""

import argparse
import hashlib
import json
from pathlib import Path
import tempfile
from urllib.request import urlopen


MANIFEST = json.loads(Path(__file__).with_name('teachers.json').read_text())
DEFAULT_TEACHER_DIR = Path(__file__).resolve().parents[2] / 'pretrained' / 'mtlkd'


def ensure_teachers(problems, output_dir=DEFAULT_TEACHER_DIR):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for problem in problems:
        filename = f'{problem.lower()}100.pt'
        metadata = MANIFEST['files'][filename]
        destination = output_dir / filename
        if destination.is_file():
            digest = hashlib.sha256(destination.read_bytes()).hexdigest()
            if digest != metadata['sha256']:
                raise ValueError(f'Checksum mismatch for {destination}; move it aside before downloading again')
            paths.append(destination)
            continue
        url = ('https://raw.githubusercontent.com/CIAM-Group/MTLKD/'
               f'{MANIFEST["revision"]}/pretrained/{filename}')
        print(f'Downloading teacher {problem}: {url}', flush=True)
        temporary = None
        try:
            hasher = hashlib.sha256()
            size = 0
            with tempfile.NamedTemporaryFile(dir=output_dir, prefix=filename + '.',
                                             suffix='.part', delete=False) as file:
                temporary = Path(file.name)
                with urlopen(url, timeout=60) as response:
                    for block in iter(lambda: response.read(1024 * 1024), b''):
                        file.write(block)
                        hasher.update(block)
                        size += len(block)
            if size != metadata['size_bytes'] or hasher.hexdigest() != metadata['sha256']:
                raise ValueError(f'Teacher download failed integrity check: {url}')
            temporary.replace(destination)
            print(f'Cached {destination} ({size / 1024**2:.1f} MiB)', flush=True)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        paths.append(destination)
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output_dir', type=Path, default=DEFAULT_TEACHER_DIR)
    parser.add_argument('--problems', nargs='+',
                        default=['CVRP', 'OVRP', 'VRPB', 'VRPL', 'VRPTW', 'OVRPTW'],
                        choices=['CVRP', 'OVRP', 'VRPB', 'VRPL', 'VRPTW', 'OVRPTW'])
    args = parser.parse_args()
    ensure_teachers(args.problems, args.output_dir)


if __name__ == '__main__':
    main()
