from __future__ import annotations

import argparse
import json
import sqlite3
import unicodedata
from pathlib import Path
from typing import Iterable


PATH_SEPARATOR = " > "


def normalize_category(value: str) -> str:
    """Return a stable key for case-insensitive category lookup."""
    value = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(value.split())


def _category_names(product: dict) -> list[str]:
    values = product.get("categories") or []
    if not isinstance(values, list):
        return []
    return [str(value).strip() for value in values if str(value).strip()]


def _create_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        PRAGMA foreign_keys = ON;

        CREATE TABLE metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE products (
            parent_asin TEXT PRIMARY KEY
        );

        CREATE TABLE categories (
            category_id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            normalized_name TEXT NOT NULL,
            full_path TEXT NOT NULL UNIQUE,
            depth INTEGER NOT NULL CHECK (depth >= 0),
            parent_id INTEGER REFERENCES categories(category_id),
            product_count INTEGER NOT NULL DEFAULT 0
        );

        CREATE TABLE product_categories (
            parent_asin TEXT NOT NULL REFERENCES products(parent_asin),
            category_id INTEGER NOT NULL REFERENCES categories(category_id),
            position INTEGER NOT NULL CHECK (position >= 0),
            is_leaf INTEGER NOT NULL CHECK (is_leaf IN (0, 1)),
            PRIMARY KEY (parent_asin, category_id)
        );

        CREATE INDEX categories_by_normalized_name
            ON categories(normalized_name);
        CREATE INDEX categories_by_parent
            ON categories(parent_id);
        CREATE INDEX product_categories_by_category
            ON product_categories(category_id, is_leaf, parent_asin);
        CREATE INDEX product_categories_by_product
            ON product_categories(parent_asin, position);
        """
    )


def build_category_database(
    catalog_path: str | Path,
    database_path: str | Path,
    *,
    overwrite: bool = False,
) -> dict[str, int]:
    """Build a normalized category tree and product/category lookup database."""
    catalog_path = Path(catalog_path)
    database_path = Path(database_path)
    if database_path.exists() and not overwrite:
        raise FileExistsError(
            f"{database_path} already exists; pass overwrite=True to replace it"
        )
    database_path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = database_path.with_name(database_path.name + ".building")
    if temporary_path.exists():
        temporary_path.unlink()

    connection = sqlite3.connect(temporary_path)
    product_count = 0
    uncategorized_count = 0
    category_ids: dict[str, int] = {}

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

                cursor.execute(
                    "INSERT INTO products(parent_asin) VALUES (?)", (parent_asin,)
                )
                product_count += 1
                names = _category_names(product)
                if not names:
                    uncategorized_count += 1
                    continue

                path_parts: list[str] = []
                parent_id: int | None = None
                for position, name in enumerate(names):
                    path_parts.append(name)
                    full_path = PATH_SEPARATOR.join(path_parts)
                    category_id = category_ids.get(full_path)
                    if category_id is None:
                        cursor.execute(
                            """
                            INSERT INTO categories(
                                name, normalized_name, full_path, depth, parent_id
                            ) VALUES (?, ?, ?, ?, ?)
                            """,
                            (
                                name,
                                normalize_category(name),
                                full_path,
                                position,
                                parent_id,
                            ),
                        )
                        category_id = int(cursor.lastrowid)
                        category_ids[full_path] = category_id

                    cursor.execute(
                        """
                        INSERT INTO product_categories(
                            parent_asin, category_id, position, is_leaf
                        ) VALUES (?, ?, ?, ?)
                        """,
                        (parent_asin, category_id, position, position == len(names) - 1),
                    )
                    parent_id = category_id

        cursor.execute(
            """
            UPDATE categories
            SET product_count = (
                SELECT COUNT(*)
                FROM product_categories pc
                WHERE pc.category_id = categories.category_id
            )
            """
        )
        cursor.executemany(
            "INSERT INTO metadata(key, value) VALUES (?, ?)",
            (
                ("schema_version", "1"),
                ("catalog_path", str(catalog_path)),
                ("product_count", str(product_count)),
                ("category_count", str(len(category_ids))),
                ("uncategorized_product_count", str(uncategorized_count)),
            ),
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
    return {
        "products": product_count,
        "categories": len(category_ids),
        "uncategorized_products": uncategorized_count,
    }


class CategoryIndex:
    """Read-only helpers for category and product lookup."""

    def __init__(self, database_path: str | Path = "data/category_index.sqlite3") -> None:
        uri = f"file:{Path(database_path).resolve()}?mode=ro"
        self.connection = sqlite3.connect(uri, uri=True)
        self.connection.row_factory = sqlite3.Row

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "CategoryIndex":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def find_categories(self, name: str) -> list[dict]:
        """Find all category nodes with this name, ordered by product count."""
        rows = self.connection.execute(
            """
            SELECT category_id, name, full_path, depth, product_count
            FROM categories
            WHERE normalized_name = ?
            ORDER BY product_count DESC, full_path
            """,
            (normalize_category(name),),
        ).fetchall()
        return [dict(row) for row in rows]

    def products_for_category(
        self,
        category: int | str,
        *,
        leaf_only: bool = False,
        limit: int | None = None,
    ) -> list[str]:
        """Return ASINs for a category ID or an exact full category path."""
        column = "c.category_id" if isinstance(category, int) else "c.full_path"
        sql = f"""
            SELECT pc.parent_asin
            FROM product_categories pc
            JOIN categories c ON c.category_id = pc.category_id
            WHERE {column} = ?
        """
        parameters: list[object] = [category]
        if leaf_only:
            sql += " AND pc.is_leaf = 1"
        sql += " ORDER BY pc.parent_asin"
        if limit is not None:
            if limit < 0:
                raise ValueError("limit must be non-negative")
            sql += " LIMIT ?"
            parameters.append(limit)
        return [
            str(row[0])
            for row in self.connection.execute(sql, parameters).fetchall()
        ]

    def categories_for_product(self, parent_asin: str) -> list[dict]:
        """Return the ordered category path for one product."""
        rows = self.connection.execute(
            """
            SELECT c.category_id, c.name, c.full_path, c.depth, pc.is_leaf
            FROM product_categories pc
            JOIN categories c ON c.category_id = pc.category_id
            WHERE pc.parent_asin = ?
            ORDER BY pc.position
            """,
            (parent_asin,),
        ).fetchall()
        return [dict(row) for row in rows]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the product category SQLite index")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--output", default="data/category_index.sqlite3")
    parser.add_argument("--force", action="store_true", help="replace an existing index")
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    counts = build_category_database(args.catalog, args.output, overwrite=args.force)
    print(
        f"Built {args.output}: {counts['products']} products, "
        f"{counts['categories']} category nodes, "
        f"{counts['uncategorized_products']} uncategorized products"
    )


if __name__ == "__main__":
    main()
