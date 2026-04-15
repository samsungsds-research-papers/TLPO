from functools import partial
from typing import Dict, List


def process_results(doc: dict, results: List[str]) -> Dict[str, int]:
    results = {
        "none": 0.0,
    }

    return results


def process_docs(dataset, subject):
    return dataset.filter(lambda x: x["language"] == subject)


process_ko = partial(process_docs, subject="ko")
process_zh = partial(process_docs, subject="zh")
process_ar = partial(process_docs, subject="ar")
process_es = partial(process_docs, subject="es")
process_fr = partial(process_docs, subject="fr")
process_ja = partial(process_docs, subject="ja")
process_it = partial(process_docs, subject="it")