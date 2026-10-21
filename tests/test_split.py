"""The per-fold CSV schema: identity first, every column named once."""
import csv
from pathlib import Path

from deepshift.data.split import REQUIRED_IDENTITY, route_labelset_masters

EXTRA_BY_SOURCE = {"af": ["plddt", "ca_shift"], "nmr": ["ca_shift", "n_shift"]}


def write_masters(root: Path) -> tuple[list[str], dict]:
    """Two source masters carrying every identity column and a few extras."""
    paths, fold_map = [], {}
    for index, (source, extras) in enumerate(EXTRA_BY_SOURCE.items()):
        columns = [*REQUIRED_IDENTITY, "residue_code", "struct_res_id",
                   "mismatch_type", "accession", "chain", *extras]
        path = root / f"structure_data_{source}.csv"
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            for residue in (1, 2):
                label_set = f"ls_{source}"
                row = {column: f"{column}-{source}-{residue}" for column in columns}
                row.update(label_set_id=label_set, comp_index_id=residue,
                           source=source, mismatch_type="match")
                writer.writerow(row)
        fold_map[f"ls_{source}"] = index + 1
        paths.append(str(path))
    return paths, fold_map


def test_fold_csv_header_names_every_column_once(tmp_path):
    masters, fold_map = write_masters(tmp_path)
    out = tmp_path / "folds"
    route_labelset_masters(masters, fold_map, str(out))
    for fold in (1, 2):
        path = out / f"structure_data_unified_fold_{fold}.csv"
        with path.open(newline="") as stream:
            header = next(csv.reader(stream))
        assert len(set(header)) == len(header)
        assert header[:1 + len(REQUIRED_IDENTITY)] == ["split", *REQUIRED_IDENTITY]
