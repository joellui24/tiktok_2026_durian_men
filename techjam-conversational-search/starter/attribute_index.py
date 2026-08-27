from __future__ import annotations

import argparse
import json
import re
import sqlite3
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence


ASK_ATTRIBUTES = (
    "category",
    "material",
    "color",
    "size",
    "style",
    "brand",
    "budget",
    "feature",
    "use_case",
    "other",
)
MATERIALS = (
    "cotton",
    "polyester",
    "nylon",
    "leather",
    "wool",
    "spandex",
    "silk",
    "rayon",
    "fabric",
)
COLORS = (
    "black",
    "white",
    "blue",
    "red",
    "pink",
    "green",
    "brown",
    "gray",
    "grey",
    "purple",
    "yellow",
    "orange",
)
MATERIAL_RE = re.compile(r"\b(" + "|".join(MATERIALS) + r")\b", re.I)
COLOR_RE = re.compile(r"\b(" + "|".join(COLORS) + r")\b", re.I)
PATH_SEPARATOR = " > "


def normalize_value(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(value.split())


def _clean(value: object, limit: int = 180) -> str:
    return re.sub(r"\s+", " ", str(value)).strip(" -;,.\t\n")[:limit].rstrip()


def _flatten(value: object) -> list[str]:
    if isinstance(value, dict):
        return [
            f"{key}: {item}"
            for key, item in value.items()
            if item not in (None, "", [])
        ]
    if isinstance(value, list):
        return [str(item) for item in value if item not in (None, "")]
    return [str(value)] if value not in (None, "") else []


def _searchable_text(product: dict) -> str:
    fields = ("title", "features", "details", "description", "categories", "store")
    return " ".join(part for field in fields for part in _flatten(product.get(field)))


def classify_constraint(value: str) -> str:
    """Mirror the local evaluator's ask_attribute classification policy."""
    lowered = value.lower()
    if "budget" in lowered or re.search(r"(?:\$|<=|under)\s*\d", lowered):
        return "budget"
    if any(material in lowered for material in MATERIALS):
        return "material"
    if any(word in lowered for word in ("color", *COLORS)):
        return "color"
    if any(word in lowered for word in ("size", "sizing", "width", "wide", "narrow")):
        return "size"
    if any(word in lowered for word in ("department", "style", "fit", "sleeve", "neck")):
        return "style"
    if any(word in lowered for word in ("hiking", "running", "gym", "winter", "outdoor", "work")):
        return "use_case"
    return "feature"


def attribute_values(product: dict) -> dict[str, list[str]]:
    """Extract canonical catalog values plus simulator-compatible constraints."""
    result: dict[str, list[str]] = {attribute: [] for attribute in ASK_ATTRIBUTES}

    categories = [
        _clean(value) for value in product.get("categories") or [] if _clean(value)
    ]
    for position, name in enumerate(categories):
        result["category"].append(name)
        result["category"].append(PATH_SEPARATOR.join(categories[: position + 1]))

    store = _clean(product.get("store") or "")
    if store:
        result["brand"].append(store)
    details = product.get("details")
    if isinstance(details, dict):
        for key, value in details.items():
            if "brand" in str(key).lower() or "manufacturer" in str(key).lower():
                cleaned = _clean(value)
                if cleaned:
                    result["brand"].append(cleaned)

    price = product.get("price")
    if isinstance(price, (int, float)) and not isinstance(price, bool):
        result["budget"].append(f"{float(price):.2f}")

    corpus = _searchable_text(product)
    result["material"].extend(match.group(1).lower() for match in MATERIAL_RE.finditer(corpus))
    result["color"].extend(match.group(1).lower() for match in COLOR_RE.finditer(corpus))

    # customer_reply only considers the first four intent-card constraints. Keep
    # the same ordering so this index reflects what the simulator can disclose.
    candidates = [*_flatten(product.get("features")), *_flatten(details)]
    material = MATERIAL_RE.search(corpus)
    color = COLOR_RE.search(corpus)
    if material:
        candidates.insert(0, material.group(1).lower())
    if color:
        candidates.insert(1, f"color: {color.group(1).lower()}")
    if price not in (None, ""):
        candidates.append(f"budget around ${price}")
    constraints = list(
        dict.fromkeys(cleaned for item in candidates if (cleaned := _clean(item)))
    )[:4]
    if not constraints:
        fallback = _clean(product.get("title") or "product")
        constraints = [fallback]

    for constraint in constraints:
        result[classify_constraint(constraint)].append(constraint)
        result["other"].append(constraint)

    for attribute, values in result.items():
        result[attribute] = list(dict.fromkeys(value for value in values if value))
    return result


def _create_schema(connection: sqlite3.Connection) -> None:
    allowed = ", ".join(f"'{attribute}'" for attribute in ASK_ATTRIBUTES)
    connection.executescript(
        f"""
        PRAGMA foreign_keys = ON;

        CREATE TABLE metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE products (
            parent_asin TEXT PRIMARY KEY,
            price REAL
        );

        CREATE TABLE attribute_values (
            attribute TEXT NOT NULL CHECK (attribute IN ({allowed})),
            normalized_value TEXT NOT NULL,
            display_value TEXT NOT NULL,
            parent_asin TEXT NOT NULL REFERENCES products(parent_asin),
            ordinal INTEGER NOT NULL,
            PRIMARY KEY (attribute, normalized_value, parent_asin)
        ) WITHOUT ROWID;

        CREATE INDEX attribute_values_by_product
            ON attribute_values(parent_asin, attribute, ordinal);
        CREATE INDEX products_by_price
            ON products(price, parent_asin);
        """
    )
    for attribute in ASK_ATTRIBUTES:
        connection.execute(
            f"""
            CREATE VIEW {attribute}_values AS
            SELECT normalized_value, display_value, parent_asin, ordinal
            FROM attribute_values
            WHERE attribute = '{attribute}'
            """
        )


def build_attribute_database(
    catalog_path: str | Path,
    database_path: str | Path,
    *,
    overwrite: bool = False,
) -> dict[str, int]:
    catalog_path = Path(catalog_path)
    database_path = Path(database_path)
    if database_path.exists() and not overwrite:
        raise FileExistsError(
            f"{database_path} already exists; pass overwrite=True to replace it"
        )
    database_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = database_path.with_name(database_path.name + ".building")
    temporary_path.unlink(missing_ok=True)

    connection = sqlite3.connect(temporary_path)
    product_count = 0
    try:
        _create_schema(connection)
        cursor = connection.cursor()
        with catalog_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                product = json.loads(line)
                parent_asin = str(product.get("parent_asin", "")).strip()
                if not parent_asin:
                    raise ValueError(f"missing parent_asin on catalog line {line_number}")
                price = product.get("price")
                numeric_price = (
                    float(price)
                    if isinstance(price, (int, float)) and not isinstance(price, bool)
                    else None
                )
                cursor.execute(
                    "INSERT INTO products(parent_asin, price) VALUES (?, ?)",
                    (parent_asin, numeric_price),
                )
                product_count += 1

                rows: list[tuple[str, str, str, str, int]] = []
                for attribute, values in attribute_values(product).items():
                    for ordinal, display_value in enumerate(values):
                        rows.append(
                            (
                                attribute,
                                normalize_value(display_value),
                                display_value,
                                parent_asin,
                                ordinal,
                            )
                        )
                cursor.executemany(
                    """
                    INSERT OR IGNORE INTO attribute_values(
                        attribute, normalized_value, display_value, parent_asin, ordinal
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    rows,
                )

        stored_counts = {
            str(attribute): int(row_count)
            for attribute, row_count in cursor.execute(
                "SELECT attribute, COUNT(*) FROM attribute_values GROUP BY attribute"
            )
        }
        metadata = {
            "schema_version": "1",
            "catalog_path": str(catalog_path),
            "product_count": str(product_count),
            **{
                f"{attribute}_row_count": str(stored_counts.get(attribute, 0))
                for attribute in ASK_ATTRIBUTES
            },
        }
        cursor.executemany(
            "INSERT INTO metadata(key, value) VALUES (?, ?)", metadata.items()
        )
        connection.commit()
        cursor.execute("PRAGMA optimize")
    except Exception:
        connection.close()
        temporary_path.unlink(missing_ok=True)
        raise
    else:
        connection.close()

    temporary_path.replace(database_path)
    return {"products": product_count, **stored_counts}


class AttributeIndex:
    def __init__(self, database_path: str | Path = "data/attribute_index.sqlite3") -> None:
        uri = f"file:{Path(database_path).resolve()}?mode=ro"
        self.connection = sqlite3.connect(uri, uri=True)

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "AttributeIndex":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @staticmethod
    def _validate_attribute(attribute: str) -> None:
        if attribute not in ASK_ATTRIBUTES:
            raise ValueError(f"attribute must be one of: {', '.join(ASK_ATTRIBUTES)}")

    def products_for_value(
        self, attribute: str, value: str, *, limit: int | None = None
    ) -> list[str]:
        self._validate_attribute(attribute)
        sql = """
            SELECT parent_asin
            FROM attribute_values
            WHERE attribute = ? AND normalized_value = ?
            ORDER BY parent_asin
        """
        parameters: list[object] = [attribute, normalize_value(value)]
        if limit is not None:
            if limit < 0:
                raise ValueError("limit must be non-negative")
            sql += " LIMIT ?"
            parameters.append(limit)
        return [str(row[0]) for row in self.connection.execute(sql, parameters)]

    def values_for_product(self, parent_asin: str, attribute: str) -> list[str]:
        self._validate_attribute(attribute)
        return [
            str(row[0])
            for row in self.connection.execute(
                """
                SELECT display_value
                FROM attribute_values
                WHERE parent_asin = ? AND attribute = ?
                ORDER BY ordinal
                """,
                (parent_asin, attribute),
            )
        ]

    def products_in_budget(
        self,
        maximum: float,
        *,
        minimum: float = 0,
        limit: int | None = None,
    ) -> list[str]:
        sql = """
            SELECT parent_asin FROM products
            WHERE price BETWEEN ? AND ?
            ORDER BY price, parent_asin
        """
        parameters: list[object] = [minimum, maximum]
        if limit is not None:
            if limit < 0:
                raise ValueError("limit must be non-negative")
            sql += " LIMIT ?"
            parameters.append(limit)
        return [str(row[0]) for row in self.connection.execute(sql, parameters)]

    def filter_products(
        self,
        filters: dict[str, str | Sequence[str]] | None = None,
        *,
        minimum_price: float | None = None,
        maximum_price: float | None = None,
        limit: int | None = None,
    ) -> list[str]:
        """Return only product IDs that survive all supplied filters.

        Different attributes are combined with AND. Multiple values for the
        same attribute are alternatives and are combined with OR.
        """
        sql = "SELECT p.parent_asin FROM products p WHERE 1 = 1"
        parameters: list[object] = []

        if minimum_price is not None:
            sql += " AND p.price >= ?"
            parameters.append(float(minimum_price))
        if maximum_price is not None:
            sql += " AND p.price < ?"
            parameters.append(float(maximum_price))

        for attribute, requested_values in (filters or {}).items():
            self._validate_attribute(attribute)
            if isinstance(requested_values, str):
                values = [requested_values]
            else:
                values = list(requested_values)
            normalized_values = list(
                dict.fromkeys(normalize_value(value) for value in values if value.strip())
            )
            if not normalized_values:
                return []
            placeholders = ", ".join("?" for _ in normalized_values)
            sql += f"""
                AND EXISTS (
                    SELECT 1
                    FROM attribute_values av
                    WHERE av.parent_asin = p.parent_asin
                      AND av.attribute = ?
                      AND av.normalized_value IN ({placeholders})
                )
            """
            parameters.append(attribute)
            parameters.extend(normalized_values)

        sql += " ORDER BY p.parent_asin"
        if limit is not None:
            if limit < 0:
                raise ValueError("limit must be non-negative")
            sql += " LIMIT ?"
            parameters.append(limit)
        return [str(row[0]) for row in self.connection.execute(sql, parameters)]

    def load_hashmap(self, attribute: str) -> dict[str, tuple[str, ...]]:
        """Load exact value -> ASINs for average O(1) in-process lookup."""
        self._validate_attribute(attribute)
        result: defaultdict[str, list[str]] = defaultdict(list)
        for value, parent_asin in self.connection.execute(
            """
            SELECT normalized_value, parent_asin
            FROM attribute_values
            WHERE attribute = ?
            ORDER BY normalized_value, parent_asin
            """,
            (attribute,),
        ):
            result[str(value)].append(str(parent_asin))
        return {value: tuple(asins) for value, asins in result.items()}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the ask_attribute SQLite index")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--output", default="data/attribute_index.sqlite3")
    parser.add_argument("--force", action="store_true", help="replace an existing index")
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    counts = build_attribute_database(args.catalog, args.output, overwrite=args.force)
    print(f"Built {args.output} for {counts.pop('products')} products")
    print("Rows by attribute: " + ", ".join(f"{key}={value}" for key, value in counts.items()))


if __name__ == "__main__":
    main()
