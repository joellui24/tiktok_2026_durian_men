from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from evaluator.local_evaluator import coarse_category as evaluator_coarse_category
from starter.category_index import (
    CategoryIndex,
    build_category_database,
    coarse_category,
)


def _write_catalog(path: Path) -> None:
    products = [
        {
            "parent_asin": "A1",
            "categories": [
                "Clothing, Shoes & Jewelry",
                "Women",
                "Clothing",
                "Tops, Tees & Blouses",
                "Tunics",
            ],
        },
        {
            "parent_asin": "A2",
            "categories": [
                "Clothing, Shoes & Jewelry",
                "Women",
                "Tops, Tees & Blouses",
                "Tunics",
            ],
        },
        {
            "parent_asin": "A3",
            "categories": [
                "Clothing, Shoes & Jewelry",
                "Men",
                "Accessories",
                "Belts",
            ],
        },
    ]
    path.write_text(
        "".join(json.dumps(product) + "\n" for product in products),
        encoding="utf-8",
    )


class CategoryIndexTest(unittest.TestCase):
    def test_coarse_category_matches_evaluator(self) -> None:
        values = [
            "Clothing, Shoes & Jewelry",
            "Women",
            "Clothing",
            "Tops, Tees & Blouses",
            "Tunics",
        ]
        self.assertEqual(coarse_category(values), evaluator_coarse_category(values))
        self.assertEqual(coarse_category(values), "Tees & Blouses Tunics")

    def test_products_table_stores_coarse_category(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temporary_path = Path(directory)
            catalog_path = temporary_path / "catalog.jsonl"
            database_path = temporary_path / "category.sqlite3"
            _write_catalog(catalog_path)

            counts = build_category_database(catalog_path, database_path)

            self.assertEqual(counts["products"], 3)
            self.assertEqual(counts["coarse_categories"], 2)
            with CategoryIndex(database_path) as index:
                self.assertEqual(
                    index.products_for_coarse_category("Tees & Blouses Tunics"),
                    ["A1", "A2"],
                )
                self.assertEqual(
                    index.coarse_category_for_product("A3"),
                    "Accessories Belts",
                )
                self.assertIsNone(index.coarse_category_for_product("missing"))


if __name__ == "__main__":
    unittest.main()
