from lit_nlp.api import dataset as lit_dataset
from lit_nlp.api import types as lit_types


import tensorflow_datasets as tfds


def load_tfds(*args, **kw):
    """Load from TFDS."""
    # Materialize to NumPy arrays.
    # This also ensures compatibility with TF1.x non-eager mode, which doesn't
    # support direct iteration over a tf.data.Dataset.
    return list(
        tfds.as_numpy(tfds.load(*args, download=True, try_gcs=True, **kw)))


class ToxicityData(lit_dataset.Dataset):
    """Jigsaw toxicity dataset; see https://www.tensorflow.org/datasets/catalog/wikipedia_toxicity_subtypes."""

    LABELS = ["non-toxic", "toxic"]

    def __init__(self, split="test", max_seq_len=500):
        """Dataset constructor, loads the data into memory."""
        raw_examples = load_tfds("wikipedia_toxicity_subtypes", split=split)
        self._examples = []  # populate this with data records
        for record in raw_examples:
            self._examples.append({
                "sentence": record["text"].decode("utf-8"),
                "label": self.LABELS[int(record["toxicity"])],
                "identity_attack": bool(int(record["identity_attack"])),
                "insult": bool(int(record["insult"])),
                "obscene": bool(int(record["obscene"])),
                "severe_toxicity": bool(int(record["severe_toxicity"])),
                "threat": bool(int(record["threat"]))
            })

    def spec(self) -> lit_types.Spec:
        """Dataset spec, which should match the model"s input_spec()."""
        return {
            "sentence": lit_types.TextSegment(),
            "label": lit_types.CategoryLabel(vocab=self.LABELS),
            "identity_attack": lit_types.Boolean(),
            "insult": lit_types.Boolean(),
            "obscene": lit_types.Boolean(),
            "severe_toxicity": lit_types.Boolean(),
            "threat": lit_types.Boolean()
        }
