import json

import pytest

from telecom.config import ROOT
from telecom.evaluate import check_case


CASES = json.loads((ROOT / "data/evaluation_cases.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("case", CASES, ids=[case["id"] for case in CASES])
def test_offline_evaluation_case(case, assistant, client):
    assert isinstance(check_case(case, assistant, client), dict)


def test_case_catalogue_covers_all_faq_topics():
    assert len(CASES) == len({case["id"] for case in CASES}) == 38
    documents = json.loads((ROOT / "data/faqs.json").read_text(encoding="utf-8"))
    covered = {case["expected"] for case in CASES if case["kind"] == "retrieval"}
    assert covered == {f"faq:{document['id']}:" for document in documents}


def test_unknown_evaluation_kind_is_rejected(assistant, client):
    with pytest.raises(ValueError, match="Unknown evaluation kind"):
        check_case({"kind": "not_a_real_case"}, assistant, client)
