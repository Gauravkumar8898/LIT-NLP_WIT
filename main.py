# from lit_nlp.examples.datasets.classification import
from lit_nlp.examples.models.glue_models import ToxicityModel
from src.lit_nlp_.dataset import ToxicityData
from src.lit_nlp_.model import ToxicityModel

from lit_nlp.examples.datasets.classification import IMDBData


from collections.abc import Sequence
import sys
from typing import Optional

from absl import app
from absl import flags
from absl import logging

from lit_nlp import dev_server
from lit_nlp import server_flags
# from lit_nlp.examples.datasets import classification
# from lit_nlp.examples.models import glue_models

# NOTE: additional flags defined in server_flags.py

FLAGS = flags.FLAGS

FLAGS.set_default("development_demo", True)

_MODEL_PATH = flags.DEFINE_string(
    "model_path",
    "https://storage.googleapis.com/what-if-tool-resources/lit-models/toxicity.tar.gz",
    "Path to saved model (from transformers library).",
)
_MAX_EXAMPLES = flags.DEFINE_integer(
    "max_examples", 1000, "Maximum number of examples to load into LIT. ")


def get_wsgi_app() -> Optional[dev_server.LitServerType]:
    """Returns a LitApp instance for consumption by gunicorn."""
    FLAGS.set_default("server_type", "external")
    FLAGS.set_default("demo_mode", True)
    # Parse flags without calling app.run(main), to avoid conflict with
    # gunicorn command line flags.
    unused = flags.FLAGS(sys.argv, known_only=True)
    if unused:
        logging.info(
            "toxicity_demo:get_wsgi_app() called with unused args: %s", unused
        )
    return main([])


def main(argv: Sequence[str]) -> Optional[dev_server.LitServerType]:
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    model_path = _MODEL_PATH.value
    logging.info("Working directory: %s", model_path)

    # Load our trained model.
    models = {"toxicity": ToxicityModel(model_path)}
    datasets = {"toxicity_test": ToxicityData("test")}

    # Truncate datasets if --max_examples is set.
    for name in datasets:
        logging.info("Dataset: '%s' with %d examples", name, len(datasets[name]))
        datasets[name] = datasets[name].slice[:_MAX_EXAMPLES.value]
        logging.info("  truncated to %d examples", len(datasets[name]))

    # Start the LIT server. See server_flags.py for server options.
    lit_demo = dev_server.Server(models, datasets, **server_flags.get_flags())
    return lit_demo.serve()


if __name__ == "__main__":
    app.run(main)

