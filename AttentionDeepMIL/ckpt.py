import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import torch


def ckpt_path(epoch=None):
    path = Path('checkpoint')
    path.mkdir(exist_ok=True)
    path = path / 'current.json'
    if epoch is None:
        if path.exists():
            with path.open() as fp:
                epoch = json.load(fp)['epoch']
        else:
            return
    else:
        with path.open('w') as fp:
            json.dump({'epoch': epoch}, fp)
    return path.parent / f'{epoch}.dat'


def save_ckpt(model, optimizer, epoch):
    def do_save(fp):
        torch.save({
            'epoch': epoch,
            'model': model,
            'optimizer': optimizer.state_dict(),
        }, fp)
    # always save as single-GPU setting
    if torch.cuda.device_count() > 1:
        model = model.module
    do_save(ckpt_path(epoch))


def load_ckpt(filepath=None):
    """ Load checkpoint file

        Parameters:
        optimizer: load parameter into optimizer from checkpoint, if not null
        filepath: filepath of checkpoint(s), otherwise lookup checkpoint/current.json

        Returns:
        Raise error if specific checkpoint file not found
        Otherwise checkpoint as SimpleNamespace Object
    """
    if filepath:
        filepath = Path(filepath)
        if not filepath.exists():
            print("Aborted: checkpoint not found!", filepath)
            exit(-2)
    else:
        filepath = ckpt_path()
        if not filepath or not filepath.exists():
            return
    print("Loading checkpoint '{}'".format(filepath))
    if torch.cuda.is_available():
        # Load all tensors onto previous state
        checkpoint = torch.load(filepath)
    else:
        # Load all tensors onto the CPU
        checkpoint = torch.load(filepath, map_location=lambda storage, loc: storage)
    return SimpleNamespace(**checkpoint)