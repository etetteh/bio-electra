# coding=utf-8
# Copyright 2020 The TensorFlow Datasets Authors and the HuggingFace Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
""" """


import csv
import os
import textwrap

import numpy as np

import datasets


_TRAINING_FILE = "train.tsv"
_DEV_FILE = "dev.tsv"
_TEST_FILE = "test.tsv"


class BioNREConfig(datasets.BuilderConfig):
    """BuilderConfig for BioNRE."""

    def __init__(
        self,
        text_features,
        label_column,
        data_dir,
        label_classes=None,
        process_label=lambda x: x,
        **kwargs,
    ):
        """BuilderConfig for BioNRE.
        Args:
          text_features: `dict[string, string]`, map from the name of the feature
            dict for each text field to the name of the column in the tsv file
          label_column: `string`, name of the column in the tsv file corresponding
            to the label
          data_dir: `string`, the path to the folder containing the tsv files in the
            downloaded zip
          label_classes: `list[string]`, the list of classes if the label is
            categorical. If not provided, then the label will be of type
            `datasets.Value('float32')`.
          process_label: `Function[string, any]`, function  taking in the raw value
            of the label and processing it to the form required by the label feature
          **kwargs: keyword arguments forwarded to super.
        """
        super(BioNREConfig, self).__init__(version=datasets.Version("1.0.0", ""), **kwargs)
        self.text_features = text_features
        self.label_column = label_column
        self.label_classes = label_classes
        self.data_dir = data_dir
        self.process_label = process_label


class BioNRE(datasets.GeneratorBasedBuilder):
    """ """

    BUILDER_CONFIGS = [
        BioNREConfig(
            name="chemprot",
            description=textwrap.dedent(
                """ """
            ),  # pylint: disable=line-too-long
            text_features={"sentence1": "sentence1"},
            label_classes=["CPR:3", "CPR:4", "CPR:5", "CPR:6", "CPR:9", "false"],
            label_column="label",
            data_dir="chemprot/",
        ),
        BioNREConfig(
            name="ddi",
            description=textwrap.dedent(
                """ """
            ),  # pylint: disable=line-too-long
            text_features={"sentence1": "sentence1"},
            label_classes=["DDI-advise", "DDI-effect", "DDI-int", "DDI-mechanism", 'DDI-false'],
            label_column="label",
            data_dir="ddi/",
        ),
        BioNREConfig(
            name="gad",
            description=textwrap.dedent(
                """ """
            ),  # pylint: disable=line-too-long
            text_features={"sentence1": "sentence1"},
            label_classes=["0", "1"],
            label_column="label",
            data_dir="gad/",
        ),
        
    ]

    def _info(self):
        features = {text_feature: datasets.Value("string") for text_feature in self.config.text_features.keys()}
        if self.config.label_classes:
            features["label"] = datasets.features.ClassLabel(names=self.config.label_classes)
        else:
            features["label"] = datasets.Value("float32")
        features["idx"] = datasets.Value("int32")
        return datasets.DatasetInfo(
            features=datasets.Features(features),
            homepage="",
        )

    def _split_generators(self, dl_manager):
        dest = f"./downstream/{self.config.name}"
            
        data_files = {
            "train": os.path.join(f"{dest}", _TRAINING_FILE),
            "dev": os.path.join(f"{dest}", _DEV_FILE),
            "test": os.path.join(f"{dest}", _TEST_FILE),
        }
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": data_files["train"], "split": "train"}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"filepath": data_files["dev"], "split": "dev"}),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepath": data_files["test"], "split": "test"}),
        ]

    def _generate_examples(self, filepath, split):
        process_label = self.config.process_label
        label_classes = self.config.label_classes

        with open(filepath, encoding="utf8") as f:
            reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
            for n, row in enumerate(reader):
                example = {feat: row[col] for feat, col in self.config.text_features.items()}
                example["idx"] = n

                if self.config.label_column in row:
                    label = row[self.config.label_column]
                    # For some tasks, the label is represented as 0 and 1 in the tsv
                    # files and needs to be cast to integer to work with the feature.
                    if label_classes and label not in label_classes:
                        label = int(label) if label else None
                    example["label"] = process_label(label)
                else:
                    example["label"] = process_label(-1)

                # Filter out corrupted rows.
                for value in example.values():
                    if value is None:
                        break
                else:
                    yield example["idx"], example

    