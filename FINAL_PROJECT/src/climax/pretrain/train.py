# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
print(currentdir)
parentdir = os.path.dirname(currentdir)
print(parentdir)
parentdir = os.path.dirname(parentdir)
print(parentdir)
sys.path.insert(0, parentdir) 

from pytorch_lightning.cli import LightningCLI

from climax.pretrain.datamodule import MultiSourceDataModule
from climax.pretrain.module import PretrainModule


def main():
    # Initialize Lightning with the model and data modules, and instruct it to parse the config yml
    print("Initializing Lightning!!!")
    cli = LightningCLI(
        model_class=PretrainModule,
        datamodule_class=MultiSourceDataModule,
        seed_everything_default=42,
        save_config_overwrite=True,
        run=False,
        auto_registry=True,
        parser_kwargs={"parser_mode": "omegaconf", "error_handler": None},
    )
    print("DONE LIGHTNING!!!")
    os.makedirs(cli.trainer.default_root_dir, exist_ok=True)

    cli.model.set_lat_lon(*cli.datamodule.get_lat_lon())

    # fit() runs the training
    print("Training Executing")
    cli.trainer.fit(cli.model, datamodule=cli.datamodule)


if __name__ == "__main__":
    main()
