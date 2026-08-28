from __future__ import annotations

import json
import math
import sqlite3
import struct
from collections import defaultdict
from pathlib import Path
from typing import Iterable


MODEL_SCHEMA_VERSION = "1"
RARE_VALUE = "<rare>"
NO_ANSWER = "<no-answer>"


def default_model_path() -> Path:
    """Return the repository-local Approach 1 artifact path."""

    return Path(__file__).resolve().parents[2] / "approach 1" / "fm_model.sqlite3"


def turn_bucket(turn: int) -> str:
    if turn <= 3:
        return "early"
    if turn <= 6:
        return "middle"
    return "late"


class PortableHybridModel:
    """Standard-library inference for the offline-trained hybrid FM.

    Ranking omits context-only FM terms because they are identical for every
    product in one response.  The stored item base already contains item
    linear and item-item FM terms, leaving one context/item latent dot product
    plus the selected explicit context-item crosses.
    """

    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path)
        uri = f"file:{self.database_path.resolve()}?mode=ro"
        connection = sqlite3.connect(uri, uri=True)
        try:
            metadata = {
                str(key): str(value)
                for key, value in connection.execute(
                    "SELECT key, value FROM metadata"
                )
            }
            if metadata.get("schema_version") != MODEL_SCHEMA_VERSION:
                raise ValueError(
                    "unsupported FM artifact schema: "
                    f"{metadata.get('schema_version')!r}"
                )
            self.metadata = metadata
            self.dimension = int(metadata["dimension"])
            self.temperature = max(float(metadata.get("temperature", "1")), 1e-6)
            vector_format = f"<{self.dimension}f"

            self.context_ids: dict[str, int] = {}
            self.context_vectors: dict[int, tuple[float, ...]] = {}
            for feature_id, name, vector_blob in connection.execute(
                "SELECT feature_id, name, vector FROM context_features"
            ):
                feature_id = int(feature_id)
                self.context_ids[str(name)] = feature_id
                self.context_vectors[feature_id] = struct.unpack(
                    vector_format, bytes(vector_blob)
                )

            self.item_feature_names: dict[int, str] = {
                int(feature_id): str(name)
                for feature_id, name in connection.execute(
                    "SELECT feature_id, name FROM item_features"
                )
            }
            self.item_feature_fields: dict[int, str] = {}
            self.item_linear_weights: dict[int, float] = {}
            for feature_id, field, linear_weight in connection.execute(
                "SELECT feature_id, field, linear_weight FROM item_features"
            ):
                self.item_feature_fields[int(feature_id)] = str(field)
                self.item_linear_weights[int(feature_id)] = float(linear_weight)
            self.context_feature_fields: dict[int, str] = {
                int(feature_id): str(field)
                for feature_id, field in connection.execute(
                    "SELECT feature_id, field FROM context_features"
                )
            }
            self.product_scores: dict[
                str, tuple[float, tuple[float, ...], tuple[int, ...]]
            ] = {}
            for parent_asin, base_score, vector_blob, feature_blob in connection.execute(
                "SELECT parent_asin, base_score, vector, item_feature_ids FROM products"
            ):
                raw_features = bytes(feature_blob)
                feature_count = len(raw_features) // 4
                features = (
                    struct.unpack(f"<{feature_count}I", raw_features)
                    if feature_count
                    else ()
                )
                self.product_scores[str(parent_asin)] = (
                    float(base_score),
                    struct.unpack(vector_format, bytes(vector_blob)),
                    tuple(int(value) for value in features),
                )

            self.cross_weights: dict[int, dict[int, float]] = defaultdict(dict)
            for context_id, item_id, weight in connection.execute(
                "SELECT context_feature_id, item_feature_id, weight FROM cross_weights"
            ):
                self.cross_weights[int(context_id)][int(item_id)] = float(weight)

            ordered_replies: dict[str, list[tuple[str, str]]] = defaultdict(list)
            for parent_asin, attribute, normalized_value in connection.execute(
                """
                SELECT parent_asin, attribute, normalized_value
                FROM reply_values
                ORDER BY parent_asin, ordinal
                """
            ):
                ordered_replies[str(parent_asin)].append(
                    (str(attribute), str(normalized_value))
                )
            self.reply_values = dict(ordered_replies)
        finally:
            connection.close()

    @property
    def product_count(self) -> int:
        return len(self.product_scores)

    def matches_catalog(self, catalog_ids: Iterable[str]) -> bool:
        identifiers = set(catalog_ids)
        return len(identifiers) == self.product_count and identifiers == set(
            self.product_scores
        )

    def resolve_context_feature(self, name: str) -> int | None:
        feature_id = self.context_ids.get(name)
        if feature_id is not None:
            return feature_id
        if "=" not in name:
            return None
        prefix = name.split("=", 1)[0]
        return self.context_ids.get(f"{prefix}={RARE_VALUE}")

    def context_ids_for(self, names: Iterable[str]) -> tuple[int, ...]:
        result: list[int] = []
        seen: set[int] = set()
        for name in names:
            feature_id = self.resolve_context_feature(name)
            if feature_id is not None and feature_id not in seen:
                seen.add(feature_id)
                result.append(feature_id)
        return tuple(result)

    def score_many(
        self,
        parent_asins: Iterable[str],
        context_names: Iterable[str],
        *,
        mode: str = "hybrid",
        disabled_field_pair: tuple[str, str] | None = None,
    ) -> dict[str, float]:
        if mode not in {"linear", "fm", "hybrid"}:
            raise ValueError("mode must be linear, fm, or hybrid")
        context_ids = self.context_ids_for(context_names)
        context_vector = [0.0] * self.dimension
        for feature_id in context_ids:
            vector = self.context_vectors[feature_id]
            for position, value in enumerate(vector):
                context_vector[position] += value

        result: dict[str, float] = {}
        for parent_asin in parent_asins:
            product = self.product_scores.get(parent_asin)
            if product is None:
                continue
            base_score, item_vector, item_features = product
            if mode == "linear":
                score = sum(
                    self.item_linear_weights.get(item_id, 0.0)
                    for item_id in item_features
                )
            else:
                score = base_score + sum(
                    left * right
                    for left, right in zip(context_vector, item_vector, strict=True)
                )
            if mode == "hybrid":
                for context_id in context_ids:
                    weights = self.cross_weights.get(context_id)
                    if not weights:
                        continue
                    for item_id in item_features:
                        if disabled_field_pair == (
                            self.context_feature_fields[context_id],
                            self.item_feature_fields[item_id],
                        ):
                            continue
                        score += weights.get(item_id, 0.0)
            result[parent_asin] = score
        return result

    def rank(
        self,
        parent_asins: Iterable[str],
        context_names: Iterable[str],
        limit: int,
        *,
        mode: str = "hybrid",
        disabled_field_pair: tuple[str, str] | None = None,
    ) -> list[str]:
        scores = self.score_many(
            parent_asins,
            context_names,
            mode=mode,
            disabled_field_pair=disabled_field_pair,
        )
        return sorted(scores, key=lambda asin: (-scores[asin], asin))[:limit]

    def posterior(
        self,
        parent_asins: Iterable[str],
        context_names: Iterable[str],
        *,
        mode: str = "hybrid",
        disabled_field_pair: tuple[str, str] | None = None,
    ) -> dict[str, float]:
        scores = self.score_many(
            parent_asins,
            context_names,
            mode=mode,
            disabled_field_pair=disabled_field_pair,
        )
        if not scores:
            return {}
        maximum = max(scores.values())
        scaled = {
            asin: math.exp((score - maximum) / self.temperature)
            for asin, score in scores.items()
        }
        total = sum(scaled.values())
        if not math.isfinite(total) or total <= 0:
            uniform = 1.0 / len(scaled)
            return {asin: uniform for asin in scaled}
        return {asin: value / total for asin, value in scaled.items()}

    def predicted_reply(
        self,
        parent_asin: str,
        attribute: str,
        disclosed_values: set[str],
    ) -> tuple[str, ...]:
        matches = [
            value
            for candidate_attribute, value in self.reply_values.get(parent_asin, [])
            if value not in disclosed_values
            and (attribute == "other" or candidate_attribute == attribute)
        ][:2]
        return tuple(matches) if matches else (NO_ANSWER,)

    def describe_crosses(self, context_names: Iterable[str]) -> list[dict[str, object]]:
        """Return active explicit crosses for diagnostics and tests."""

        rows: list[dict[str, object]] = []
        for context_id in self.context_ids_for(context_names):
            context_name = next(
                name for name, feature_id in self.context_ids.items()
                if feature_id == context_id
            )
            for item_id, weight in self.cross_weights.get(context_id, {}).items():
                rows.append(
                    {
                        "context_feature": context_name,
                        "item_feature": self.item_feature_names[item_id],
                        "weight": weight,
                    }
                )
        return rows

    def cross_contributions(
        self,
        parent_asins: Iterable[str],
        context_names: Iterable[str],
    ) -> dict[tuple[str, str], dict[str, float]]:
        """Return explicit-cross score contributions grouped by field pair."""

        context_ids = self.context_ids_for(context_names)
        result: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
        for parent_asin in parent_asins:
            product = self.product_scores.get(parent_asin)
            if product is None:
                continue
            item_features = product[2]
            for context_id in context_ids:
                weights = self.cross_weights.get(context_id)
                if not weights:
                    continue
                context_field = self.context_feature_fields[context_id]
                for item_id in item_features:
                    weight = weights.get(item_id)
                    if weight is None:
                        continue
                    field_pair = (context_field, self.item_feature_fields[item_id])
                    by_product = result[field_pair]
                    by_product[parent_asin] = by_product.get(parent_asin, 0.0) + weight
        return dict(result)


def artifact_metadata(database_path: str | Path) -> dict[str, object]:
    model = PortableHybridModel(database_path)
    return {
        **model.metadata,
        "product_count_loaded": model.product_count,
        "context_feature_count_loaded": len(model.context_ids),
        "item_feature_count_loaded": len(model.item_feature_names),
        "cross_count_loaded": sum(len(values) for values in model.cross_weights.values()),
    }


def metadata_json(database_path: str | Path) -> str:
    return json.dumps(artifact_metadata(database_path), indent=2, sort_keys=True)
