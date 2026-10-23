"""Download the public OpenPI π0.5 LIBERO checkpoint, resuming complete files."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import json
import gcsfs

SOURCE = 'openpi-assets/checkpoints/pi05_libero'
DEST = Path('/data/rohan/Reward-Modeling/pi0_checkpoints/pi05_libero')


def one(remote, size):
    relative = remote.removeprefix(SOURCE + '/')
    target = DEST / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size == size:
        return relative, size, 'cached'
    partial = target.with_name(target.name + '.partial')
    if partial.exists():
        partial.unlink()
    gcsfs.GCSFileSystem(token='anon').get_file(remote, str(partial))
    if partial.stat().st_size != size:
        raise IOError(f'Incomplete download: {relative}: {partial.stat().st_size}/{size}')
    partial.replace(target)
    return relative, size, 'downloaded'


def main():
    fs = gcsfs.GCSFileSystem(token='anon')
    files = [(f, fs.info(f)['size']) for f in fs.find(SOURCE)]
    print(f'{len(files)} files, {sum(s for _, s in files) / 1e9:.2f} GB', flush=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(one, f, s) for f, s in files]
        for future in as_completed(futures):
            print(*future.result(), flush=True)
    (DEST / 'download_manifest.json').write_text(json.dumps({
        'source': f'gs://{SOURCE}/',
        'files': {f.removeprefix(SOURCE + '/'): size for f, size in files},
        'total_bytes': sum(size for _, size in files),
    }, indent=2) + '\n')

if __name__ == '__main__':
    main()
