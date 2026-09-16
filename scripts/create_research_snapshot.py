"""Capture a byte-preserving research snapshot before release migration."""

import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def git(*arguments):
    return subprocess.check_output(
        ['git', '-c', f'safe.directory={ROOT.as_posix()}', *arguments], cwd=ROOT,
        text=True, encoding='utf-8',
    )


def main():
    head = git('rev-parse', 'HEAD').strip()
    destination = ROOT / 'archive/research' / f'pre_final_{head[:12]}'
    if destination.exists():
        raise FileExistsError('Snapshot already exists; immutable snapshots cannot be overwritten')
    tracked = set(git('ls-files').splitlines())
    files = []
    for directory in ('src', 'tests', 'results', 'weka/src'):
        files.extend(p for p in (ROOT / directory).rglob('*') if p.is_file()
                     and '__pycache__' not in p.parts and p.suffix not in {'.pyc', '.class'})
    files.extend(ROOT / p for p in ('run_all.py', 'CMakeLists.txt', 'requirements.txt',
                                   'README.md', '.gitignore', 'weka/pom.xml', 'data/raw/README.md',
                                   '.vscode/tasks.json') if (ROOT / p).is_file())
    entries = []
    for path in sorted(set(files)):
        relative = path.relative_to(ROOT).as_posix()
        ignored = subprocess.run(['git', '-c', f'safe.directory={ROOT.as_posix()}',
                                  'check-ignore', '-q', relative], cwd=ROOT).returncode == 0
        archived = destination / 'snapshot' / relative
        archived.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, archived)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert hashlib.sha256(archived.read_bytes()).hexdigest() == digest
        category = 'required-evidence' if relative.startswith('results/') else 'research-source'
        entries.append({'old_relative_path': relative,
                        'archived_relative_path': 'snapshot/' + relative,
                        'sha256': digest, 'size': path.stat().st_size, 'category': category,
                        'git_state': 'tracked' if relative in tracked else (
                            'ignored research evidence' if ignored else 'untracked'),
                        'role': 'Historical reproduction; never an active runtime dependency',
                        'limitations': 'Raw dataset, numerical packages and native toolchains supplied separately'})
    manifest = {'snapshot_id': destination.name, 'git_head': head,
                'git_status_before_snapshot': git('status', '--short'), 'files': entries,
                'excluded': ['.git', '.venv', 'build products', 'weka/target', 'data/raw dataset',
                             'data/processed', 'cache', 'empty verification directory'],
                'inventory': {'active_candidates': ['src/custom_knn/classifier.py', 'src/data',
                     'src/preprocessing/pipeline.py', 'src/model_quality_v2/features.py',
                     'src/model_quality_v2/transforms.py', 'src/model_quality_v2/custom_v2_knn.py',
                     'src/cpp_knn', 'weka/src'],
                     'research_only': ['src/tuning', 'src/model_quality', 'src/model_quality_v2 search',
                                       'old src/benchmarking', 'results before migration'],
                     'historical_sources_missing': 'No V1-V7 sources invented; only locally available files captured'}}
    (destination / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    (destination / 'README.md').write_text(
        '# Research snapshot\n\nThis snapshot contains the current local pre-release tree, including '
        'available historical evidence. Raw SHA-256 hashes refer to the original bytes.\n\n'
        'Restore snapshot/ into a separate disposable working directory, provide the raw dataset '
        'separately, install requirements.txt there, and run tests there. Do not run historical '
        'search or scripts in this repository or inside archive/. No historical expensive search '
        'is required to reproduce the final model. See ARCHIVE_NOTES.md beside this file.\n', encoding='utf-8')
    (destination / 'ARCHIVE_NOTES.md').write_text(
        '# Historical limitations\n\nOriginal documents and false legacy_test_evaluated flags are '
        'preserved verbatim. They describe historical selection bookkeeping, not untouched test '
        'data. Some early reports used Stage-5 candidate evidence instead of retained evidence; '
        'the migration characterizes the actual retained configuration. Historical timing scopes '
        'and k19 uniform results must not be substituted for final k101 weighted timings. '
        'Previously inspected test results are continuity evidence. The primary balanced_low_fp '
        'mode was adopted by the user after reviewing earlier results.\n', encoding='utf-8')
    print(json.dumps({'snapshot': destination.relative_to(ROOT).as_posix(),
                      'files': len(entries), 'bytes': sum(e['size'] for e in entries),
                      'verified': True}))


if __name__ == '__main__':
    main()
