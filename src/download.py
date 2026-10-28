"""Obtém o ZIP oficial e verifica o mesmo SHA-256 usado nos experimentos."""
import argparse
import hashlib
from pathlib import Path
import urllib.request

URL = 'https://huggingface.co/datasets/garythung/trashnet/resolve/main/dataset-resized.zip?download=true'
EXPECTED_SHA256 = 'c060e8abfe5d6de0578ca15be1ed8ad0794a865d333c3473d53d1d9ad6e38b8c'

def checksum(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('.'))
    args = parser.parse_args()
    target = args.root.resolve() / 'data/raw/dataset-resized.zip'
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if checksum(target) != EXPECTED_SHA256:
            raise ValueError('ZIP existente difere da referência. Preserve-o separadamente e verifique sua origem.')
        print(f'ZIP já verificado: {target}')
        return
    temporary = target.with_suffix('.zip.part')
    try:
        request = urllib.request.Request(URL, headers={'User-Agent': 'TrashNet-course-reproducibility/1.0'})
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open('wb') as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        if checksum(temporary) != EXPECTED_SHA256:
            raise ValueError('O download não corresponde ao SHA-256 do dataset usado neste estudo.')
        temporary.replace(target)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    print(f'Download concluído e SHA-256 verificado: {target}')

if __name__ == '__main__':
    main()
