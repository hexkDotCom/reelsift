"""Moments: candidates from different cameras that show the same thing at the same time.

A candidate shows the same moment as a picked clip when it comes from another camera, overlaps the
pick in real time (with a tolerance that grows when either clock is uncertain) and looks alike.
The relation is checked pair by pair against what is already picked rather than grouped up front,
because overlapping windows would otherwise chain whole recordings into one "moment".
Footage only one camera caught is never affected.
"""

from dataclasses import dataclass

import numpy as np

from .tripdata import TripData

TOLERANCE_S = 2.0
UNSURE_TOLERANCE_S = 30.0
UNSURE_CONFIDENCE = 0.5
SIMILAR_COS = 0.75


@dataclass
class MomentIndex:
    start: np.ndarray
    """UTC seconds of each candidate's core start (NaN when the recording isn't placed)."""
    end: np.ndarray
    camera: np.ndarray
    tolerance: np.ndarray
    embedding: np.ndarray
    cameras: int

    def same_moment(self, index: int) -> np.ndarray:
        """Boolean mask of candidates showing the same moment as candidate `index`."""
        if self.cameras < 2 or np.isnan(self.start[index]):
            return np.zeros(len(self.start), bool)
        tol = np.maximum(self.tolerance, self.tolerance[index])
        with np.errstate(invalid="ignore"):
            overlap = (self.start < self.end[index] + tol) & (self.start[index] - tol < self.end)
        similar = self.embedding @ self.embedding[index] >= SIMILAR_COS
        return overlap & similar & (self.camera != self.camera[index])


def index(data: TripData, embeddings: np.ndarray) -> MomentIndex:
    c = data.candidates
    n = len(c)
    start = np.full(n, np.nan)
    end = np.full(n, np.nan)
    camera = np.zeros(n, np.int64)
    tolerance = np.zeros(n)
    for i in range(n):
        r = data.recordings[int(c.recording[i])]
        camera[i] = r.camera_id
        tolerance[i] = UNSURE_TOLERANCE_S if r.clock_confidence < UNSURE_CONFIDENCE else TOLERANCE_S
        if r.utc_start is not None:
            start[i] = r.utc_start + c.core_start[i]
            end[i] = r.utc_start + c.core_end[i]
    return MomentIndex(start, end, camera, tolerance, embeddings, len(set(camera.tolist())))
