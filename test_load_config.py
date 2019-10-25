import yaml
import argparse
from pathlib import Path
from argparse import Namespace

# Let's load the yaml file here
with open("config.yaml", "r") as f:
    config = yaml.load(f)

print(config)


args = Namespace(
    frequency_cutoff=config["frequency_cutoff"],
    model_state_file=config["model_state_file"],
    file_csv=config["file_csv"],
    save_dir=config["save_dir"],
    vectorizer_file=config["vectorizer_file"],
    batch_size=config["batch_size"],
    early_stopping_criteria=config["early_stopping_criteria"],
    learning_rate=config["learning_rate"],
    num_epochs=config["num_epochs"],
    seed=config["seed"],
    catch_keyboard_interrupt=config["catch_keyboard_interrupt"],
    cuda=config["cuda"],
    expand_filepaths_to_save_dir=config["expand_filepaths_to_save_dir"],
    reload_from_files=config["reload_from_files"],
)
print()
print(args)
