"""Single MMseqs2 execution core for deposition and modeled sequences."""

import os
import shutil
import subprocess
import tempfile
from collections import defaultdict, deque

# Pinned wide open so identity and coverage are the only criteria: MMseqs' 1e-3 default
# acts as a third filter and drops exact-subsequence pairs (see run_all_vs_all).
EVALUE = 10000


def resolve_mmseqs():
    candidates = (
        os.environ.get('MMSEQS_BIN'),
        os.environ.get('MMSEQS'),
        shutil.which('mmseqs'),
    )
    for candidate in candidates:
        if candidate and (os.path.isfile(candidate) or shutil.which(candidate)):
            return candidate
    from .external import ensure_mmseqs
    tool_dir = os.environ.get('SHIFT_TOOL_DIR', os.path.join(os.getcwd(), '.deepshift-tools'))
    return str(ensure_mmseqs(tool_dir))


def pairwise_edges_to_components(sequence_ids, edges):
    """Return multi-member connected components of an explicit pairwise graph.
    ``easy-cluster`` partitions are not edge-complete -- a qualifying pair can land in
    different greedy representatives -- so fold grouping must close the pairwise search
    edges, not reinterpret representative-to-member rows as all qualifying relationships."""
    nodes = set(sequence_ids)
    adjacency = defaultdict(set)
    for left, right in edges:
        if left not in nodes or right not in nodes:
            raise ValueError(
                f'pairwise MMseqs edge references an unknown id: {left!r}, '
                f'{right!r}')
        if left == right:
            continue
        adjacency[left].add(right)
        adjacency[right].add(left)

    components = []
    unseen = set(adjacency)
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        members = {start}
        queue = deque([start])
        while queue:
            node = queue.popleft()
            for neighbor in adjacency[node]:
                if neighbor not in members:
                    members.add(neighbor)
                    unseen.discard(neighbor)
                    queue.append(neighbor)
        if len(members) > 1:
            components.append(sorted(members))
    return sorted(components, key=lambda members: (members[0], members))


def run_all_vs_all(sequences, *, identity=0.80, coverage=0.90,
                   cov_mode=0, threads=4, timeout=1800, binary=None):
    """Run an edge-complete MMseqs all-vs-all search: canonical unordered ``(left, right)``
    pairs for every non-self hit passing MMseqs' identity/coverage contract. Result cap =
    full DB size (default ``--max-seqs 300`` would truncate large near-identical families).
    ``-e`` is pinned wide open too and load-bearing: at the ``1e-3`` default it became a
    third criterion and dropped exact-subsequence pairs, since a short peptide's long->short
    ``--cov-mode 1 -c 0.9`` hit scores just above that cutoff in a database of thousands.
    At the default ``-e`` an entity build finds no cross-fold edges at all, while the
    wide ``-e 10000`` finds them, including across a held-out fold; ``-s`` does not
    recover them.
    """
    if not sequences:
        return set()
    invalid = [identifier for identifier in sequences
               if not identifier or any(c.isspace() for c in identifier)]
    if invalid:
        raise ValueError(
            'MMseqs FASTA ids must be nonblank and whitespace-free; sample: '
            f'{invalid[:3]}')

    binary = binary or resolve_mmseqs()
    with tempfile.TemporaryDirectory() as tmpdir:
        fasta = os.path.join(tmpdir, 'sequences.fasta')
        with open(fasta, 'w') as handle:
            for identifier in sorted(sequences):
                handle.write(f'>{identifier}\n{sequences[identifier]}\n')
        result_path = os.path.join(tmpdir, 'all_vs_all.tsv')
        scratch = os.path.join(tmpdir, 'tmp')
        os.makedirs(scratch, exist_ok=True)
        result_cap = max(1, len(sequences))
        command = [
            binary, 'easy-search', fasta, fasta, result_path, scratch,
            '--min-seq-id', str(identity), '-c', str(coverage),
            '--cov-mode', str(cov_mode), '--max-seqs', str(result_cap),
            '--max-accept', str(result_cap), '--threads', str(threads),
            '-e', str(EVALUE), '--format-output', 'query,target',
        ]
        result = subprocess.run(command, capture_output=True, text=True,
                                timeout=timeout)
        if result.returncode != 0:
            raise RuntimeError(
                f'MMseqs2 failed ({result.returncode}): {result.stderr[-2000:]}')

        known = set(sequences)
        edges = set()
        with open(result_path) as handle:
            for line_number, line in enumerate(handle, 1):
                fields = line.rstrip('\n').split('\t')
                if len(fields) != 2:
                    raise RuntimeError(
                        f'MMseqs2 emitted malformed all-vs-all row '
                        f'{line_number}: {line.rstrip()!r}')
                left, right = fields
                if left not in known or right not in known:
                    raise RuntimeError(
                        'MMseqs2 all-vs-all output references an unknown FASTA '
                        f'id: {left!r}, {right!r}')
                if left != right:
                    edges.add(tuple(sorted((left, right))))
        return edges


def run_segment_search(queries, targets, *, identity=0.90, threads=4, timeout=1800,
                       binary=None):
    """Every ``(query, target)`` hit at or above ``identity`` with NO coverage floor, as
    ``{(query, target): {'identity', 'aligned', 'query_coverage', 'target_coverage'}}``.

    The coverage floor of the other searches is what lets a long, near-identical segment
    of a longer construct slip past them; this search reports the aligned length instead,
    so the caller can apply a minimum segment length. ``-e`` and the result caps are wide
    open as elsewhere, so a hit is admitted on identity alone.
    """
    if not queries or not targets:
        return {}
    binary = binary or resolve_mmseqs()
    with tempfile.TemporaryDirectory() as tmpdir:
        paths = {}
        for name, sequences in (('query', queries), ('target', targets)):
            paths[name] = os.path.join(tmpdir, f'{name}.fasta')
            with open(paths[name], 'w') as handle:
                for identifier in sorted(sequences):
                    handle.write(f'>{identifier}\n{sequences[identifier]}\n')
        result_path = os.path.join(tmpdir, 'segments.tsv')
        scratch = os.path.join(tmpdir, 'tmp')
        os.makedirs(scratch, exist_ok=True)
        result_cap = max(1, len(targets))
        command = [
            binary, 'easy-search', paths['query'], paths['target'], result_path,
            scratch, '--min-seq-id', str(identity), '-c', '0', '--cov-mode', '0',
            '--max-seqs', str(result_cap), '--max-accept', str(result_cap),
            '--threads', str(threads), '-e', str(EVALUE),
            '--format-output', 'query,target,fident,alnlen,qcov,tcov',
        ]
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
        if result.returncode != 0:
            raise RuntimeError(
                f'MMseqs2 failed ({result.returncode}): {result.stderr[-2000:]}')
        hits = {}
        with open(result_path) as handle:
            for line_number, line in enumerate(handle, 1):
                fields = line.rstrip('\n').split('\t')
                if len(fields) != 6:
                    raise RuntimeError(
                        f'MMseqs2 emitted malformed search row {line_number}: '
                        f'{line.rstrip()!r}')
                query, target, fident, alnlen, qcov, tcov = fields
                if query not in queries or target not in targets:
                    raise RuntimeError(
                        f'MMseqs2 search output references an unknown FASTA id: '
                        f'{query!r}, {target!r}')
                hit = {'identity': float(fident), 'aligned': int(alnlen),
                       'query_coverage': float(qcov), 'target_coverage': float(tcov)}
                current = hits.get((query, target))
                if current is None or (hit['aligned'], hit['identity']) > (
                        current['aligned'], current['identity']):
                    hits[(query, target)] = hit
        return hits


def run_best_hit_identity(queries, targets, *, coverage=0.50, cov_mode=2,
                          threads=4, timeout=1800, binary=None):
    """Best hit of every query sequence in ``targets``.

    Returns ``({query id: {'target', 'identity', 'query_coverage'}}, command)``; a query
    with no hit at or above the coverage floor is absent. Identity is MMseqs' ``fident``
    over the aligned region. ``-e`` stays wide open and the result caps are the target
    database size, so a hit is admitted on identity and coverage alone.
    """
    if not queries or not targets:
        raise ValueError('a best-hit search needs both a query and a target set')
    binary = binary or resolve_mmseqs()
    with tempfile.TemporaryDirectory() as tmpdir:
        paths = {}
        for name, sequences in (('query', queries), ('target', targets)):
            paths[name] = os.path.join(tmpdir, f'{name}.fasta')
            with open(paths[name], 'w') as handle:
                for identifier in sorted(sequences):
                    handle.write(f'>{identifier}\n{sequences[identifier]}\n')
        result_path = os.path.join(tmpdir, 'best_hit.tsv')
        scratch = os.path.join(tmpdir, 'tmp')
        os.makedirs(scratch, exist_ok=True)
        result_cap = max(1, len(targets))
        command = [
            binary, 'easy-search', paths['query'], paths['target'], result_path,
            scratch, '--min-seq-id', '0', '-c', str(coverage),
            '--cov-mode', str(cov_mode), '--max-seqs', str(result_cap),
            '--max-accept', str(result_cap), '--threads', str(threads),
            '-e', str(EVALUE), '--format-output', 'query,target,fident,qcov',
        ]
        result = subprocess.run(command, capture_output=True, text=True,
                                timeout=timeout)
        if result.returncode != 0:
            raise RuntimeError(
                f'MMseqs2 failed ({result.returncode}): {result.stderr[-2000:]}')
        best = {}
        with open(result_path) as handle:
            for line_number, line in enumerate(handle, 1):
                fields = line.rstrip('\n').split('\t')
                if len(fields) != 4:
                    raise RuntimeError(
                        f'MMseqs2 emitted malformed search row {line_number}: '
                        f'{line.rstrip()!r}')
                query, target, fident, qcov = fields
                if query not in queries or target not in targets:
                    raise RuntimeError(
                        f'MMseqs2 search output references an unknown FASTA id: '
                        f'{query!r}, {target!r}')
                hit = {'target': target, 'identity': float(fident),
                       'query_coverage': float(qcov)}
                # Ties are broken on coverage and then on the target id, so the reported
                # hit does not depend on the order MMseqs wrote its rows in.
                current = best.get(query)
                if current is None or (
                        (hit['identity'], hit['query_coverage'], hit['target'])
                        > (current['identity'], current['query_coverage'],
                           current['target'])):
                    best[query] = hit
        return best, ' '.join(command)
