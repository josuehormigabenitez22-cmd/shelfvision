"""Run detection, barcode reading and linking on an image, each stage only when needed and only once.

Orchestrator holds the models, shared by every image. Scene holds one image and caches what each
stage computes for it: OWLv2's image features, the objects, the barcodes and MoGe's normal map, and
SAM's image embedding when segmentation is on. A repeated or related query then reuses that work,
and Scene.timings tells what each stage cost.
"""

from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import replace
from functools import cached_property
from pathlib import Path
from time import perf_counter
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image

from shelfvision import timing
from shelfvision.barcodes import read_barcodes
from shelfvision.datatypes import ALL_OBJECTS, Barcode, DetectedObject, Query, clean_name, normalize_query
from shelfvision.detector import ImageFeatures, ObjectDetector
from shelfvision.devices import AUTO
from shelfvision.draw import render_barcodes, render_detections
from shelfvision.export import build_report
from shelfvision.images import load_image
from shelfvision.normals import NormalEstimator, SurfaceMap, arrow_tip, with_normals
from shelfvision.relations import Relation, assign, merge_objects, relate
from shelfvision.segmenter import ImageEmbedding, Segmenter


class Orchestrator:
    """The models shared by every image, each loaded the first time a task needs it."""

    def __init__(
        self,
        detector: ObjectDetector | None = None,
        estimator: NormalEstimator | None = None,
        threads: int | None = None,
        device: str = AUTO,
        segment: bool = False,
        segmenter: Segmenter | None = None,
    ) -> None:
        """threads sets torch's CPU threads; device is "auto", "cpu" or "cuda", see devices.py

        segment=True adds the outline of every detected object, from SAM 2.1, and linking then ties
        a code to the object whose outline holds it. Off by default: only boxes, and SAM is never loaded.
        """
        # torch keeps one thread pool per process, so this applies to every model and every image
        if threads:
            torch.set_num_threads(threads)
        self.detector = detector or ObjectDetector(device)
        self.estimator = estimator or NormalEstimator(device)
        self.segmenter = (segmenter or Segmenter(device)) if segment else None

    def open(self, path: Path) -> "Scene":
        return Scene(Path(path), self.detector, self.estimator, self.segmenter)


class Scene:
    """One image and everything computed for it so far"""

    def __init__(
        self,
        path: Path,
        detector: ObjectDetector,
        estimator: NormalEstimator,
        segmenter: Segmenter | None = None,
    ) -> None:
        self.path = path
        self._detector = detector
        self._estimator = estimator
        self._segmenter = segmenter
        self._queries: dict[Query, tuple[DetectedObject, ...]] = {}
        self.timings: dict[str, float] = {}

    @contextmanager
    def _timed(self, stage: str) -> Iterator[None]:
        """Add the seconds spent inside the block to the stage's total"""
        start = perf_counter()
        yield
        seconds = perf_counter() - start
        self.timings[stage] = self.timings.get(stage, 0.0) + seconds
        timing.record(stage, seconds)  # also for the run's execution time report

    @cached_property
    def image(self) -> Image.Image:
        with self._timed("load image"):
            return load_image(self.path)

    # Each stage takes what it needs before starting its clock, so that no stage counts another
    @cached_property
    def _features(self) -> ImageFeatures:
        image = self.image
        with self._timed("encode image (OWLv2)"):
            return self._detector.encode_image(image)

    @cached_property
    def objects(self) -> tuple[DetectedObject, ...]:
        """Every object in the image, named with the vocabulary: the "all" mode of detection"""
        features = self._features
        with self._timed("find all objects"):
            found = self._detector.detect_all(features)
        return self._outlined(found)

    @cached_property
    def _embedding(self) -> ImageEmbedding:
        image, segmenter = self.image, self._segmenter
        assert segmenter is not None, "only called with segmentation on"
        with self._timed("encode image (SAM)"):
            return segmenter.encode_image(image)

    def _outlined(self, objects: tuple[DetectedObject, ...]) -> tuple[DetectedObject, ...]:
        """The objects with their outlines when segmentation is on; unchanged when it is off"""
        if self._segmenter is None or not objects:
            return objects
        embedding = self._embedding
        with self._timed("segment objects (SAM)"):
            outlines = self._segmenter.outlines(embedding, [item.box for item in objects])
        return tuple(replace(item, outline=line) for item, line in zip(objects, outlines, strict=True))

    @cached_property
    def _barcodes(self) -> tuple[Barcode, ...]:
        """Every code read in the image, without normals: linking needs no normal map"""
        gray = cv2.cvtColor(np.asarray(self.image), cv2.COLOR_RGB2GRAY)
        with self._timed("read barcodes"):
            return tuple(read_barcodes(gray))

    @cached_property
    def _surface(self) -> SurfaceMap:
        image = self.image
        with self._timed("estimate normals (MoGe)"):
            return self._estimator.estimate(image)

    def detect(self, query: str | Iterable[str]) -> tuple[DetectedObject, ...]:
        """Detection: the boxes of the requested objects, or of every object for "all" """
        names = normalize_query(query)
        if names == ALL_OBJECTS:
            return self.objects
        if names not in self._queries:
            features = self._features
            with self._timed("find requested objects"):
                found = self._detector.detect_names(features, names)
            self._queries[names] = self._outlined(found)
        return self._queries[names]

    def _with_context(self, names: tuple[str, ...]) -> tuple[DetectedObject, ...]:
        return merge_objects(self.detect(names), self.objects)

    def _barcodes_of(self, names: Query) -> tuple[Barcode, ...]:
        """Every code for "all"; otherwise the codes whose owner carries one of the names"""
        if names == ALL_OBJECTS:
            return self._barcodes
        if not names:
            return ()
        owners = assign(self._barcodes, self._with_context(names))
        return tuple(code for code, holder in owners if holder and clean_name(holder.name) in names)

    def read_barcodes(self, query: str | Iterable[str]) -> tuple[Barcode, ...]:
        """Barcodes: the codes on the requested objects, or every code for "all", with their normals
        MoGe runs only when there is at least one code to give a normal to.
        """
        selected = self._barcodes_of(normalize_query(query))
        return with_normals(selected, self._surface) if selected else ()

    def relate(self, query: str) -> Relation:
        """Linking: object names for a barcode value, or barcode values for an object name"""
        # For a barcode value every object counts. For a name the query also runs on its own, so the
        # object comes back with the exact name the user asked for
        if any(code.value == query for code in self._barcodes):
            return relate(query, self.objects, self._barcodes)
        names = tuple(filter(None, [clean_name(query)]))
        return relate(query, self._with_context(names), self._barcodes)

    def draw_detections(self, query: str | Iterable[str]) -> Image.Image:
        """The detection image."""
        return render_detections(self.image, self.detect(query), query)

    def draw_barcodes(self, query: str | Iterable[str]) -> Image.Image:
        """The barcode image: outline, value and normal arrow of each code"""
        barcodes = self.read_barcodes(query)
        tips = [arrow_tip(code, self._surface.intrinsics) for code in barcodes]
        return render_barcodes(self.image, barcodes, tips, query)

    def report(self, query: str | Iterable[str]) -> dict[str, Any]:
        """Detections and barcodes for the query as plain data, with the object each code is printed on"""
        names = normalize_query(query)
        barcodes = self.read_barcodes(names)
        # The rest of the objects stay in play too. Without them, a code printed on the mug could end up
        # given to a requested box that just happens to overlap it
        context = self.objects if names == ALL_OBJECTS else self._with_context(names)
        owners = assign(barcodes, context)
        return build_report(self.path.name, self.image.size, names, self.detect(names), owners, self.timings)
