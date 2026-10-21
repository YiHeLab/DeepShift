"""Canonical structure parsing shared by the data pipeline and live inference.

One code path reads records, resolves alternate locations by occupancy, keys residues
by (chain, auth seq id, insertion code, comp id) so 10 and 10A never merge, and
canonicalises MSE -> MET (SE -> SD) identically for training and inference.

The cache and the atom-cloud builder are keyed by integer auth seq ids, so
``to_int_resid`` projects onto that contract, keeping the lowest insertion code
per number. Full per-insertion-code residues survive in ``canonical_residues``.
"""
from __future__ import annotations

import numpy as np

from .domain import AA_3_TO_1, RESIDUE_TO_IDX


def is_hydrogen(atom_name: str) -> bool:
    """True for PDB hydrogen/deuterium names, including old digit prefixes."""
    name = atom_name.strip()
    if not name:
        return False
    if name[0] in {"H", "D"}:
        return True
    return len(name) >= 2 and name[0].isdigit() and name[1] in {"H", "D"}


def _kabsch(mobile: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cm, ct = mobile.mean(0), target.mean(0)
    u, _, vt = np.linalg.svd((mobile - cm).T @ (target - ct))
    d = np.linalg.det(u @ vt)
    rot = u @ np.diag([1.0, 1.0, d]) @ vt
    return rot, ct - cm @ rot


def _resolve_residue(key, entry) -> dict:
    """Collapse a residue's raw records into one occupancy-coherent conformer: the
    altloc letter with the highest summed occupancy (ties to the lower letter, so the
    result is independent of file order), plus every blank-altloc atom."""
    chain, seq, icode = key
    records = entry["_records"]
    alt_occ: dict[str, float] = {}
    for altloc, _name, occ, _xyz in records:
        if altloc:
            alt_occ[altloc] = alt_occ.get(altloc, 0.0) + occ
    best = sorted(alt_occ, key=lambda a: (-alt_occ[a], a))[0] if alt_occ else None

    atoms: dict[str, np.ndarray] = {}
    for altloc, name, _occ, xyz in records:
        if altloc == "" or altloc == best:
            atoms[name] = xyz
    return {
        "residue_name": entry["residue_name"],
        "atoms": atoms,
        "auth_chain_id": chain,
        "auth_seq_id": seq,
        "insertion_code": icode,
    }


def parse_models(path, chain=None, keep_partner_chains=False) -> list[dict]:
    """Parse every MODEL of a PDB into residues keyed by (chain, seq, icode).

    ``chain`` None or '_' keeps all chains. ``keep_partner_chains`` also
    retains every other chain's records, so the target's 8 Å cloud can see partner
    context; ``partner_protein_residues`` does the standard-protein/solvent filtering
    downstream.
    """
    models: list[dict] = []
    current: dict = {}

    with open(path) as stream:
        for line in stream:
            record = line[:6].strip()
            if record == "MODEL":
                current = {}
                continue
            if record == "ENDMDL":
                if current:
                    models.append({k: _resolve_residue(k, v) for k, v in current.items()})
                current = {}
                continue
            if record not in ("ATOM", "HETATM"):
                continue
            if len(line.rstrip("\n")) < 22:
                # Too short to hold a chain id at col 22. Skipping keeps the build
                # alive; the coverage/identity gates catch a truncated structure.
                continue
            ch = line[21].strip()
            if (chain is not None and chain != "_" and ch != chain
                    and not keep_partner_chains):
                continue
            try:
                seq = int(line[22:26])
                xyz = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
            except (ValueError, IndexError):
                continue
            icode = line[26].strip() if len(line) > 26 else ""
            altloc = line[16].strip() if len(line) > 16 else ""
            try:
                occ = float(line[54:60])
            except (ValueError, IndexError):
                occ = 1.0
            key = (ch, seq, icode)
            entry = current.setdefault(
                key, {"residue_name": line[17:20].strip(), "_records": []})
            entry["_records"].append((altloc, line[12:16].strip(), occ, xyz))

    # Keep a final model even when a readable PDB omits ENDMDL, including the last
    # model of a multi-model ensemble.
    if current:
        models.append({k: _resolve_residue(k, v) for k, v in current.items()})
    return models


def ca_temperature_factors(path, chain: str = "") -> dict:
    """``{auth seq id: (temperature factor, residue name)}`` from one chain's CA records.

    ``parse_models`` drops the temperature-factor column, because the cache is built from
    coordinates alone. AlphaFold writes its per-residue confidence there, and a
    crystallographic B factor lives there too, so a reader that wants that column asks for
    it by name and gets the first model only.
    """
    out = {}
    with open(path, errors="ignore") as stream:
        for line in stream:
            if line.startswith("ENDMDL"):
                break
            if not line.startswith("ATOM") or line[12:16].strip() != "CA":
                continue
            if chain and chain != "_" and line[21] != chain:
                continue
            try:
                out[int(line[22:26])] = (float(line[60:66]), line[17:20].strip())
            except ValueError:
                continue
    return out


def representative_model_index(models: list[dict]) -> int:
    """Index of the NMR model closest to the ensemble median (single model = 0).

    Superimposes every model onto model 1 on shared CAs (Kabsch) and takes the lowest
    RMSD to the per-atom median; key arity is irrelevant. This is the single source of the
    "representative NMR model" choice, so ``structure_instance_id`` names the exact
    conformer whose coordinates are cached.
    """
    if not models:
        return 0
    if len(models) == 1:
        return 0
    ref_keys = [k for k in sorted(models[0]) if "CA" in models[0][k].get("atoms", {})]
    if len(ref_keys) < 10:
        return 0
    common = set(ref_keys)
    for model in models[1:]:
        common &= {k for k in model if "CA" in model[k].get("atoms", {})}
    keys = sorted(common)
    if len(keys) < 10:
        return 0
    coords = np.asarray([[m[k]["atoms"]["CA"] for k in keys] for m in models], dtype=np.float64)
    ref = coords[0]
    aligned = [ref]
    for item in coords[1:]:
        rot, trans = _kabsch(item, ref)
        aligned.append(item @ rot + trans)
    aligned = np.asarray(aligned)
    median = np.median(aligned, axis=0)
    rmsds = [np.sqrt(np.mean(np.sum((a - median) ** 2, axis=1))) for a in aligned]
    return int(np.argmin(rmsds))


def select_representative_model(models: list[dict]) -> dict | None:
    """The representative NMR model dict (single-model = itself)."""
    if not models:
        return None
    return models[representative_model_index(models)]


def _seq_icode(key, info) -> tuple[int, str]:
    """Author seq id + insertion code, from info if present else the key."""
    seq = info.get("auth_seq_id")
    if seq is not None:
        return int(seq), info.get("insertion_code", "")
    if isinstance(key, tuple):
        return int(key[1]), str(key[2]) if len(key) >= 3 else ""
    return int(key), ""


def canonical_residues(model, keep_nonstandard: bool = False,
                       drop_hydrogens: bool = False) -> dict:
    """Canonical preprocessing shared by training and inference.

    Returns residues keyed by (auth_seq_id, insertion_code) so 10 and 10A stay
    distinct, with MSE mapped to MET (SE -> SD). Nonstandard residues are dropped
    unless ``keep_nonstandard`` tags them is_standard=False for neighbour context.
    Accepts models keyed by (chain, seq[, icode]) or bare int.
    """
    out: dict = {}
    for key in sorted(model, key=lambda k: (str(k),)):
        info = model[key]
        seq, icode = _seq_icode(key, info)
        residue_name = info["residue_name"]
        atoms = info.get("atoms", {})
        atoms = {n: xyz for n, xyz in atoms.items()
                 if not (drop_hydrogens and is_hydrogen(n))}
        if residue_name == "MSE":
            residue_name = "MET"
            if "SE" in atoms:
                atoms["SD"] = atoms.pop("SE")
            standard = True
        else:
            standard = residue_name in AA_3_TO_1 or residue_name in RESIDUE_TO_IDX
        if not standard and not keep_nonstandard:
            continue
        out.setdefault((seq, icode), {
            "residue_name": residue_name,
            "is_standard": standard,
            "atoms": atoms,
            "auth_chain_id": info.get("auth_chain_id", key[0] if isinstance(key, tuple) else ""),
            "auth_seq_id": seq,
            "insertion_code": icode,
        })
    return out


def to_int_resid(canonical: dict) -> dict:
    """Project canonical (seq, icode) residues to the int-keyed contract.

    The int-keyed cache cannot represent insertion codes, so a seq number carrying
    several keeps the lowest (blank before 'A'); merging them would make a chimera.
    """
    out: dict = {}
    for seq, icode in sorted(canonical):
        out.setdefault(int(seq), canonical[(seq, icode)])
    return out


def project_model_to_chain_seq(model: dict) -> dict:
    """Collapse a (chain, seq, icode) model to the (chain, seq) contract.

    Keeps the lowest insertion code per (chain, seq), so an integer-keyed parser gets
    one coherent residue and never a merge of 10 and 10A. Siblings are dropped here;
    ``canonical_residues`` keeps them.
    """
    out: dict = {}
    for key in sorted(model):
        info = model[key]
        chain = info.get("auth_chain_id", key[0])
        seq = int(info.get("auth_seq_id", key[1]))
        out.setdefault((chain, seq), {
            "residue_name": info["residue_name"],
            "atoms": info["atoms"],
            "auth_chain_id": chain,
            "auth_seq_id": seq,
            "insertion_code": info.get("insertion_code", ""),
        })
    return out


def partner_protein_residues(model, target_chain, *, keep_nonstandard: bool = False,
                             drop_hydrogens: bool = True) -> dict:
    """Standard partner-protein residues of every chain except ``target_chain``.

    ``model`` comes from ``parse_models(..., keep_partner_chains=True)``. Partners get
    the same canonicalisation as the target and keep the full (chain, seq, icode) key,
    so they collide neither with the target's integer ids nor with each other. Solvent,
    ligands and nucleic acids carry no standard amino acid and so fall out in
    ``canonical_residues``, leaving partner protein context for the 8 Å cloud.
    """
    out: dict = {}
    for key in sorted(model, key=lambda k: (str(k),)):
        info = model[key]
        chain = info.get("auth_chain_id", key[0] if isinstance(key, tuple) else "")
        if chain == target_chain:
            continue
        canon = canonical_residues(
            {key: info}, keep_nonstandard=keep_nonstandard, drop_hydrogens=drop_hydrogens)
        for (seq, icode), residue in canon.items():
            out[(chain, seq, icode)] = residue
    return out


def parse_structure(path, chain: str = "A", keep_nonstandard: bool = False) -> dict | None:
    """Representative model -> integer-keyed residues, hydrogens dropped. None if empty."""
    model = select_representative_model(parse_models(path, chain=chain))
    if not model:
        return None
    return to_int_resid(canonical_residues(
        model, keep_nonstandard=keep_nonstandard, drop_hydrogens=True)) or None


def parse_pdb(path, chain_id=None):
    """First PDB model projected to ``(chain, author residue number)`` keys."""
    models = parse_models(path, chain=chain_id)
    return project_model_to_chain_seq(models[0]) if models else {}


def classify_atom(atom_name):
    """Return ``hydrogen`` for H/D isotopes and ``heavy`` otherwise."""
    return "hydrogen" if is_hydrogen(atom_name) else "heavy"
