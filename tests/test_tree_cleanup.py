"""Cleanup must retain real bead geometry and connected contact branches."""

import numpy as np

from pellet_support.printability import (
    dedupe_seeds,
    prune_disconnected_fill,
    supported_mask,
)


def test_duplicate_cleanup_preserves_order_and_removes_identical_spheres():
    first = (1.0, 0.0, 0.1, 0.2)
    second = (0.0, 0.0, 0.1, 0.2)
    seeds = [first, second, first, second, first]
    cleaned, removed = dedupe_seeds(seeds)
    assert cleaned == [first, second]
    assert removed == 3


def test_numerical_noise_at_symmetric_positions_does_not_duplicate_a_sphere():
    seeds = [(5.0, -5.55e-17, 4.0, 0.2), (5.0, 5.55e-17, 4.0, 0.2)]
    cleaned, removed = dedupe_seeds(seeds)
    assert cleaned == seeds[:1]
    assert removed == 1


def test_coincident_spheres_with_different_diameters_are_not_duplicates():
    # The large sphere is the bed contact and supports the upper bead. The
    # smaller sphere at its centre cannot replace either of those connections.
    seeds = [(0.0, 0.0, 0.1, 0.04), (0.0, 0.0, 0.1, 0.2),
             (0.0, 0.0, 0.25, 0.2)]
    cleaned, removed = dedupe_seeds(seeds)
    assert cleaned == seeds
    assert removed == 0
    assert supported_mask(cleaned, None)[1:].all()


def test_nearby_small_beads_can_be_essential_links():
    # The middle two are only 0.019 mm apart, but deleting either breaks the
    # sole bottom-up chain. A fixed 0.02 mm deduplication tolerance is unsafe.
    seeds = [(0.0, 0.0, z, 0.04) for z in (0.016, 0.05, 0.069, 0.102)]
    assert supported_mask(seeds, None).all()
    cleaned, removed = dedupe_seeds(seeds)
    assert cleaned == seeds
    assert removed == 0
    assert supported_mask(cleaned, None).all()


def test_duplicate_cleanup_accepts_numpy_input():
    seeds = np.asarray([(0.0, 0.0, 0.1, 0.2)] * 2)
    cleaned, removed = dedupe_seeds(seeds)
    np.testing.assert_array_equal(cleaned, seeds[:1])
    assert removed == 1


def test_fill_cleanup_preserves_connections_within_its_touch_slack():
    # Both beads belong to the same bed-connected component under the stated
    # 2% contact slack. The broad-phase search must include that 2% shell.
    seeds = [(0.0, 0.0, 0.5, 1.0), (0.0, 0.0, 1.51, 1.0)]
    cleaned, removed = prune_disconnected_fill(seeds, [], bed_z=0.0)
    assert cleaned == seeds
    assert removed == 0


def test_fill_cleanup_retains_bed_contacts_and_large_branches():
    bed = [(0.0, 0.0, 0.1, 0.2)]
    contact_branch = [(2.0, 0.0, z, 0.2) for z in (10.0, 10.18)]
    large_branch = [(4.0, 0.0, 10.0 + i * 0.18, 0.2) for i in range(9)]
    clutter = [(6.0, 0.0, 10.0, 0.2), (8.0, 0.0, 10.0, 0.2)]
    seeds = bed + contact_branch + large_branch + clutter
    cleaned, removed = prune_disconnected_fill(
        seeds, [(2.0, 0.0, 10.29)], bed_z=0.0)
    assert cleaned == bed + contact_branch + large_branch
    assert removed == 2


def test_fill_cleanup_handles_many_isolated_groups_with_one_contact_query(monkeypatch):
    # A separate mesh query per component made isolated fill especially costly.
    # Contact classification should query all candidate beads together.
    import pellet_support.printability as module

    original_tree = module.cKDTree
    queries = []

    class CountingTree:
        def __init__(self, points):
            self.tree = original_tree(points)

        def query_pairs(self, *args, **kwargs):
            return self.tree.query_pairs(*args, **kwargs)

        def query(self, points, *args, **kwargs):
            queries.append(len(points))
            return self.tree.query(points, *args, **kwargs)

    monkeypatch.setattr(module, "cKDTree", CountingTree)
    seeds = [(float(i), 0.0, 10.0, 0.2) for i in range(4096)]
    cleaned, removed = prune_disconnected_fill(
        seeds, [(0.0, 0.0, 10.1)], bed_z=0.0)
    assert cleaned == seeds[:1]
    assert removed == len(seeds) - 1
    assert queries == [len(seeds)]


def test_fill_cleanup_accepts_numpy_input():
    seeds = np.asarray([(0.0, 0.0, 0.1, 0.2)])
    cleaned, removed = prune_disconnected_fill(seeds, [], bed_z=0.0)
    np.testing.assert_array_equal(cleaned, seeds)
    assert removed == 0


def test_empty_cleanup_is_unchanged():
    assert dedupe_seeds([]) == ([], 0)
    assert prune_disconnected_fill([], [], 0.0) == ([], 0)
