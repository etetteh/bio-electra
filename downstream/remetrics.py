from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score,
    f1_score
)

import datasets


_DESCRIPTION = """Relation Extraction Custom Metrics"""

_KWARGS_DESCRIPTION = """\ 
    Args:
        predictions
        references
        normalize
        labels
        pos_label
        average
        sample_weight
        zero_division
        
    Returns:
        "accuracy": Accuracy
        "precision": Precision score
        "recall": Recall score
        "f1": F1 score
"""

_CITATION = """\
@article{scikit-learn,
  title={Scikit-learn: Machine Learning in {P}ython},
  author={Pedregosa, F. and Varoquaux, G. and Gramfort, A. and Michel, V.
         and Thirion, B. and Grisel, O. and Blondel, M. and Prettenhofer, P.
         and Weiss, R. and Dubourg, V. and Vanderplas, J. and Passos, A. and
         Cournapeau, D. and Brucher, M. and Perrot, M. and Duchesnay, E.},
  journal={Journal of Machine Learning Research},
  volume={12},
  pages={2825--2830},
  year={2011}
}
"""


@datasets.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class REMetrics(datasets.Metric):
    def _info(self):
        return datasets.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features(
                {
                    "predictions": datasets.Sequence(datasets.Value("int32")),
                    "references": datasets.Sequence(datasets.Value("int32")),
                }
                if self.config_name == "multilabel"
                else {
                    "predictions": datasets.Value("int32"),
                    "references": datasets.Value("int32"),
                }
            ),
            reference_urls=[],
        )
    
    def _compute(
            self,
            predictions,
            references,
            normalize=True,
            labels=None,
            pos_label=1,
            average="binary",
            sample_weight=None,
            zero_division="warn",
        ):
        acc = accuracy_score(
            references, 
            predictions, 
            normalize=normalize, 
            sample_weight=sample_weight
        )

        pr = precision_score(
            references,
            predictions,
            labels=labels,
            pos_label=pos_label,
            average=average,
            sample_weight=sample_weight,
            zero_division=zero_division,
        )

        re = recall_score(
            references,
            predictions,
            labels=labels,
            pos_label=pos_label,
            average=average,
            sample_weight=sample_weight,
            zero_division=zero_division,
        )

        f1 = f1_score(
            references, predictions, labels=labels, pos_label=pos_label, average=average, sample_weight=sample_weight
        )

        return {
            "accuracy": float(acc),
            "precision": float(pr) if pr.size == 1 else pr,
            "recall": float(re) if re.size == 1 else re,
            "f1": float(f1) if f1.size == 1 else f1,
        }
