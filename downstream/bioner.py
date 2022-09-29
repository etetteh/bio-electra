# coding=utf-8

# Lint as: python3

import os 
import csv 
import sys
import shutil
import datasets
from glob import glob


csv.field_size_limit(sys.maxsize)
logger = datasets.logging.get_logger(__name__)

_DESCRIPTION = """Datasets for biomedical named-entity recognition """

_URL = "http://nlp.dmis.korea.edu/projects/biobert-2020-checkpoints/datasets.tar.gz"
_TRAINING_FILE = "train.txt"
_DEV_FILE = "dev.txt"
_TEST_FILE = "test.txt"


class TSV2TXT:
    """Convert original dataset from .tsv to .txt format"""
    def __init__(self):
        pass
    
    def write_tsv_to_txt(self, src_dir, dest_dir):
        data_paths = glob(f"{src_dir}/*.tsv")
        os.makedirs(f"{dest_dir}", exist_ok=True)

        for data_path in data_paths:
            f_name, ext = data_path.split("/")[-1].split(".")
            if f_name == "train_dev":
                f_name = "train"
            elif f_name == "devel":
                f_name = "dev"
            elif f_name == "train":
                continue
            with open(f"{data_path}", 'r', encoding='utf-8') as f:
                with open(f"{dest_dir}/{f_name}.txt", 'w', encoding='utf-8') as f_out:
                    f = csv.reader(f, delimiter="\t")
                    for line in f:
                        f_out.write(" ".join(line)+"\n")
                print("wrote file to destination")
                    
                    
class BioNERConfig(datasets.BuilderConfig):
    """BuilderConfig for BioNER"""

    def __init__(self, **kwargs):
        """BuilderConfig for BioNER.

        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(BioNERConfig, self).__init__(**kwargs)


class BioNER(datasets.GeneratorBasedBuilder):
    """BioNER dataset."""

    BUILDER_CONFIGS = [
        BioNERConfig(name="bc2gm", version=datasets.Version("1.0.0"), 
                     description="BC2GM dataset (https://link.springer.com/article/10.1186/gb-2008-9-s2-s2)"),
        BioNERConfig(name="bc4chemd", version=datasets.Version("1.0.0"), 
                     description="BC4CHEMD dataset (https://jcheminf.biomedcentral.com/articles/10.1186/1758-2946-7-S1-S2)"),
        BioNERConfig(name="bc5cdr-chem", version=datasets.Version("1.0.0"), 
                     description="BC5CDR-chem dataset (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4860626/)"),
        BioNERConfig(name="bc5cdr-disease", version=datasets.Version("1.0.0"), 
                     description="BC5CDR-disease dataset (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4860626/)"),
        BioNERConfig(name="jnlpba", version=datasets.Version("1.0.0"), 
                     description="JNLPBA dataset (https://www.aclweb.org/anthology/W04-1213/)"),
        BioNERConfig(name="linnaeus", version=datasets.Version("1.0.0"), 
                     description="LINNAEUS dataset (https://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-11-85)"),
        BioNERConfig(name="ncbi-disease", version=datasets.Version("1.0.0"), 
                     description="NCBI-disease dataset (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3951655/)"),
        BioNERConfig(name="s800", version=datasets.Version("1.0.0"), 
                     description="Species-800 dataset (https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0065390)"),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "tokens": datasets.Sequence(datasets.Value("string")),
                          "ner_tags": datasets.Sequence(
                        datasets.features.ClassLabel(
                            names=[
                                "B",
                                "I",
                                "O",
                                "X"
                            ]
                        )
                    ),
                }
            ),
            supervised_keys=None
        )
 
    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        downloaded_file = dl_manager.download_and_extract(_URL)
        print(f"file downloaded {downloaded_file}")
        
        dest = f"~/{self.config.name}"
        if self.config.name == "bc2gm":
            shutil.copytree(f"{downloaded_file}/datasets/NER/BC2GM/", f"{dest}", dirs_exist_ok=True)
        elif self.config.name == "bc4chemd":
            shutil.copytree(f"{downloaded_file}/datasets/NER/BC4CHEMD/", f"{dest}", dirs_exist_ok=True)
        elif self.config.name == "bc5cdr-chem":
            shutil.copytree(f"{downloaded_file}/datasets/NER/BC5CDR-chem/", f"{dest}", dirs_exist_ok=True)
        elif self.config.name == "bc5cdr-disease":
            shutil.copytree(f"{downloaded_file}/datasets/NER/BC5CDR-disease/", f"{dest}", dirs_exist_ok=True)
        elif self.config.name == "jnlpba":
            shutil.copytree(f"{downloaded_file}/datasets/NER/JNLPBA/", f"{dest}", dirs_exist_ok=True)
        elif self.config.name == "linnaeus":
            shutil.copytree(f"{downloaded_file}/datasets/NER/linnaeus/", f"{dest}", dirs_exist_ok=True)
        elif self.config.name == "ncbi-disease":
            shutil.copytree(f"{downloaded_file}/datasets/NER/NCBI-disease/", f"{dest}", dirs_exist_ok=True)
        elif self.config.name == "s800":
            shutil.copytree(f"{downloaded_file}/datasets/NER/s800/", f"{dest}", dirs_exist_ok=True)
            
        TSV2TXT().write_tsv_to_txt(src_dir=f"{dest}", dest_dir=f"{dest}")

        files_to_remove = glob(f"{dest}/*.tsv")
        for file in files_to_remove:
            try:
                os.remove(file)
            except OSError as e:
                print(f"Error: {file} : {e.strerror}")
                
        data_files = {
            "train": os.path.join(f"{dest}", _TRAINING_FILE),
            "dev": os.path.join(f"{dest}", _DEV_FILE),
            "test": os.path.join(f"{dest}", _TEST_FILE),
        }

        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": data_files["train"]}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"filepath": data_files["dev"]}),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepath": data_files["test"]}),
        ]
        
    def _generate_examples(self, filepath):
        logger.info("⏳ Generating examples from = %s", filepath)
        with open(filepath, encoding="utf-8") as f:
            guid = 0
            tokens = []
            ner_tags = []
            for line in f:
                if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                    if tokens:
                        yield guid, {
                            "id": str(guid),
                            "tokens": tokens,
                            "ner_tags": ner_tags,
                        }
                        guid += 1
                        tokens = []
                        ner_tags = []
                else:
                    splits = line.split(" ")
                    if len(splits) == 2:
                        tokens.append(splits[0])
                        ner_tags.append(splits[1].rstrip())
            # last example
            if tokens:
                yield guid, {
                    "id": str(guid),
                    "tokens": tokens,
                    "ner_tags": ner_tags,
                }
                