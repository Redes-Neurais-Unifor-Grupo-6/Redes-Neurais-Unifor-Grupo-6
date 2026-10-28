"""Busca de taxa da MLP exclusivamente na validação, antes de abrir o teste.

Os dois pilotos não pertencem aos seis experimentos finais. Os resultados da
configuração inicial são preservados, e os pares finais reiniciam do zero.
Não execute em paralelo ao treinamento: isso prejudica a medição de custo.
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from .common import atomic_json, experiments, read_json, write_csv, utc_now
from .train import train_one

RATES = [0.0001, 0.0003]

def search(root: Path, pilot: float | None = None):
    configs = experiments(root)
    base = next(c for c in configs if c['id'] == 'mlp_128')
    search_dir = root / 'artifacts/search'
    if pilot is None and (search_dir / 'decision.json').exists():
        print('Busca já concluída; decisão e configurações preservadas.')
        return
    # A entrada interna --pilot também deve respeitar o bloqueio após o teste,
    # antes de criar diretórios, links ou iniciar um treinamento.
    if any((root / f'artifacts/runs/{c["id"]}/test_predictions.csv').exists() for c in configs):
        raise RuntimeError('O teste já foi avaliado. Não é permitido iniciar esta busca no mesmo protocolo.')
    pilot_root = search_dir / 'pilot_workspace'
    pilot_root.mkdir(parents=True, exist_ok=True)
    data_link = pilot_root / 'data'
    if not data_link.exists():
        # Link relativo continua válido ao mover a pasta completa do projeto.
        data_link.symlink_to('../../../data', target_is_directory=True)
    if pilot is not None:
        if pilot not in RATES:
            raise ValueError('Taxa fora da grade registrada.')
        config = {**base, 'learning_rate': pilot, 'id': f'mlp_lr_{pilot:.4f}'.replace('.', 'p')}
        return train_one(pilot_root, config)
    # A configuração inicial precisa estar totalmente concluída antes de pilotos.
    for config in configs:
        if not (root / f'artifacts/runs/{config["id"]}/metrics.json').exists():
            raise RuntimeError('Conclua os seis treinos iniciais antes da busca, sem avaliá-los no teste.')
    initial = read_json(root / 'artifacts/runs/mlp_128/metrics.json')
    initial_config = read_json(root / 'artifacts/runs/mlp_128/config.json')
    atomic_json(search_dir / 'initial_experiments.json', read_json(root / 'configs/experiments.json'))
    rows = [{
        'id': 'baseline_mlp_0p0010', 'learning_rate': initial_config['learning_rate'],
        'val_macro_f1': initial['val_macro_f1'], 'val_accuracy': initial['val_accuracy'],
        'val_loss': initial['val_loss'], 'best_epoch': initial['best_epoch'],
        'epochs_trained': initial['epochs_trained'], 'train_seconds': initial['train_seconds'],
    }]
    for rate in RATES:
        subprocess.run([sys.executable, '-m', 'src.search', '--root', str(root), '--pilot', str(rate)], check=True)
        identifier = f'mlp_lr_{rate:.4f}'.replace('.', 'p')
        result = read_json(pilot_root / f'artifacts/runs/{identifier}/metrics.json')
        rows.append({'id': identifier, 'learning_rate': rate, **{k: result[k] for k in ['val_macro_f1','val_accuracy','val_loss','best_epoch','epochs_trained','train_seconds']}})
    write_csv(search_dir / 'search_summary.csv', rows)
    winner = min(rows, key=lambda r: (-r['val_macro_f1'], r['val_loss'], r['id']))
    if winner['learning_rate'] != initial_config['learning_rate']:
        selection = root / 'artifacts/selection.json'
        if selection.exists():
            selection.rename(search_dir / 'initial_selection_before_search.json')
        for identifier in ['mlp_128', 'mlp_128_weighted']:
            source = root / f'artifacts/runs/{identifier}'
            destination = search_dir / 'initial_runs' / identifier
            destination.parent.mkdir(parents=True, exist_ok=True)
            source.rename(destination)
        config_file = root / 'configs/experiments.json'
        final_config = read_json(config_file)
        for config in final_config['experiments']:
            if config['architecture'] == 'mlp':
                config['learning_rate'] = winner['learning_rate']
        atomic_json(config_file, final_config)
    atomic_json(search_dir / 'decision.json', {
        'completed_at_utc': utc_now(), 'selected_learning_rate': winner['learning_rate'],
        'criterion': 'highest validation macro F1 of unweighted MLP; ties lowest validation loss then id',
        'test_used': False, 'rates': [initial_config['learning_rate'], *RATES],
        'reason': 'Initial MLP predicted almost a single class on validation. Two smaller Adam learning rates were investigated before test access.',
        'final_training': 'Both MLP experiments restarted from seed 42 with selected learning rate; CNN protocol unchanged.',
    })
    print(f"Taxa selecionada apenas pela validação: {winner['learning_rate']}. Execute src.train para concluir os modelos finais.")

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('.'))
    p.add_argument('--pilot', type=float, help=argparse.SUPPRESS)
    a = p.parse_args()
    search(a.root.resolve(), a.pilot)

if __name__ == '__main__':
    main()
