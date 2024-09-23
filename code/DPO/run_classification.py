import argparse
import yaml
from pathlib import Path
import wandb  # Import W&B

from classifier import ClassifierModel, ClassifierConfig  
from datasets import Dataset

from data_utils import compile_DatasetFrames


def main(config: ClassifierConfig):
    """Main method to train and evaluate the classifier."""

    # initialize W&B project and config tracking
    wandb.init(project="AdaParse-text-cls", config=config.dict())

    # compile the dataset frames (train/val/test) using `compile_DatasetFrames`
    df_train, df_test, df_val = compile_DatasetFrames(
        p_embeddings=config.p_embeddings_root_dir,
        p_response=config.p_response_csv_path,
        parser=config.parser,
        #f_train=config.frequency_train,
        #seed_val=config.seed_val,
        normalized=config.normalized,
        predefined_split=config.predefined_split,
        p_split_yaml_path=config.p_split_yaml_path
    )

    # convert the pandas DataFrames to HuggingFace Datasets
    dset_train = Dataset.from_pandas(df_train)
    dset_val = Dataset.from_pandas(df_val)
    dset_test = None
    if config.test_flag:
        dset_test = Dataset.from_pandas(df_test)

    # initialize the classifier
    classifier = ClassifierModel(config)

    # train the classifier
    classifier.train(dset_train, eval_data=dset_val)

    # evaluate the classifier on validation and test datasets
    val_results = classifier.evaluate(dset_val)
    wandb.log(val_results)

    # log to W&B
    if config.test_flag and dset_test is not None:
        test_results = classifier.evaluate(dset_test)
        wandb.log(test_results)  

    # complete wandb logging
    wandb.finish() 

if __name__ == "__main__":
    # parse the config file path from the command line
    parser = argparse.ArgumentParser(description="Run classification model with configuration file")
    parser.add_argument("-c", "--config", type=str, required=True, help="Path to the YAML config file")
    args = parser.parse_args()

    # load the YAML configuration file
    config_path = Path(args.config)
    assert config_path.is_file(), f"Config file {config_path} does not exist."
    # - open
    with open(config_path, "r") as file:
        config_dict = yaml.safe_load(file)

    # create a ClassifierConfig instance
    config = ClassifierConfig(**config_dict)

    # run
    main(config)
