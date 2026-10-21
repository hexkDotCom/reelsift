import numpy as np

from reelsift.evaluate import Moment, evaluate
from reelsift.select import Item, Selection

from .test_select import trip_data


def test_evaluate_counts_covered_cores():
    data = trip_data()
    wanted = [Moment(1, 10, 14, True), Moment(2, 50, 52, True), Moment(1, 90, 92, True)]
    unwanted = [Moment(3, 0, 4, False)]
    items = [
        Item(1, 9, 13, 1, 0, None, "window"),  # covers 3 of 4 core seconds
        Item(2, 51.5, 54, 2, 0, None, "window"),  # covers a quarter
        Item(3, 1, 5, 3, 0, None, "window"),  # an unwanted moment
    ]
    r = evaluate(data, Selection(30, items, 0), wanted + unwanted)
    assert (r.found, r.wanted, r.unwanted_included) == (1, 3, 1)
    assert r.candidate_recall == 1.0
    assert np.isclose(r.recall, 1 / 3)
