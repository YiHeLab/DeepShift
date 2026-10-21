"""Frame-safe, near-identity sequence alignment for structure-shift matching.

BLOSUM62 / open=-10 / extend=-0.5 (conservative gap costs); gaps are legal only in
the shorter sequence (equal-length pairs get none), so no delete/insert pair can turn
a substitution into a false match.
"""

from Bio import Align
from Bio.Align import substitution_matrices

ALIGNMENT_AMBIGUITY_MARGIN = 1.0


def frame_safe_alignments(seq1, seq2, *, score_margin=0.0):
    """Frame-safe global alignments within ``score_margin`` of best: exactly
    ``abs(len(seq1) - len(seq2))`` gaps are permitted, all in the shorter sequence
    (a positive margin, materialization-only, finds near-ties; ordinary callers use zero)."""
    aligner = Align.PairwiseAligner()
    aligner.mode = 'global'
    aligner.substitution_matrix = substitution_matrices.load('BLOSUM62')
    aligner.open_gap_score = -10
    aligner.extend_gap_score = -0.5
    if score_margin:
        # epsilon = paths within this of the optimum; +1e-6 pad includes the named boundary exactly.
        aligner.epsilon = float(score_margin) + 1e-6

    # There is no path-constraint switch, so the forbidden gap direction is dominated by
    # score: BLOSUM62 is bounded by +11/-4, and this input-sized penalty swamps every
    # possible substitution gain -- exact without -inf, which segfaults on some inputs
    # (Biopython 1.87).
    forbidden = -32.0 * (len(seq1) + len(seq2) + 1)
    # The gap scores are set through the target_*/query_* names and verified by
    # construction in the tests rather than by name, since the mapping is easy to invert.
    if len(seq1) <= len(seq2):
        # Insertion = query residue vs gap in target/seq1; required when seq1 is shorter,
        # forbidden at equal length -- so the opposite, a gap in the query, is forbidden.
        aligner.query_open_gap_score = forbidden
        aligner.query_extend_gap_score = forbidden
    if len(seq1) >= len(seq2):
        # A deletion is a target/seq1 residue opposite a gap in query/seq2.
        aligner.target_open_gap_score = forbidden
        aligner.target_extend_gap_score = forbidden

    return aligner.align(seq1, seq2)


def align_sequences(seq1, seq2):
    """Return the best frame-safe near-identity alignment, or ``None``."""
    alignments = frame_safe_alignments(seq1, seq2)
    # `alignments[0]`/`bool(alignments)` call __len__, which raises OverflowError above
    # 2**63 equally-optimal alignments (low-complexity seqs); iterate instead (O(1)).
    try:
        alignment = next(iter(alignments))
    except StopIteration:
        return None

    # Return nothing if the alignment violates the frame-safe gap contract.
    a, b = str(alignment[0]), str(alignment[1])
    if (('-' in a and '-' in b)
            or a.count('-') != max(len(seq2) - len(seq1), 0)
            or b.count('-') != max(len(seq1) - len(seq2), 0)):
        return None
    return alignment
