import unittest

from clover_diffusion.decoding import Prediction, generate_blocks, resolve_block, select_expert_coreset


class FakeRuntime:
    def __init__(self, rounds):
        self.rounds = iter(rounds)
        self.seen = []
        self.commits = []

    def preview(self, block, pending):
        self.seen.append((block, pending, tuple(self.commits)))
        return next(self.rounds)

    def commit(self, tokens):
        self.commits.append(tokens)


def predictions(*confidences):
    return [Prediction(10 + i, c) for i, c in enumerate(confidences)]


class DenoisingTests(unittest.TestCase):
    def test_threshold_parallel_and_progress_fallback(self):
        runtime = FakeRuntime([predictions(.96, .5, .95), predictions(.1, .7, .7), predictions(.2, .1, .2)])
        block, rounds = resolve_block(runtime, width=3, mask_token=99, threshold=.95)
        self.assertEqual(block, [10, 11, 12])
        self.assertEqual(rounds, 3)
        self.assertEqual(runtime.seen[1][0], (10, 99, 99))
        self.assertEqual(runtime.seen[2][0], (10, 11, 99))
        self.assertEqual(runtime.commits, [])

    def test_partial_last_block_still_resolves_full_width(self):
        runtime = FakeRuntime([predictions(1, 1, 1, 1), predictions(1, 1, 1, 1)])
        out = list(generate_blocks(runtime, max_new_tokens=6, block_size=4, mask_token=99))
        self.assertEqual([x.tokens for x in out], [(10, 11, 12, 13), (10, 11)])
        self.assertEqual(runtime.commits, [(10, 11, 12, 13)])
        self.assertEqual(len(runtime.seen[1][0]), 4)

    def test_eos_after_full_resolution_no_commit(self):
        runtime = FakeRuntime([predictions(.99, .99, .99)])
        out = list(generate_blocks(runtime, max_new_tokens=3, block_size=3, mask_token=99, eos_tokens=(11,)))
        self.assertEqual(out[0].tokens, (10, 11))
        self.assertTrue(out[0].stopped)
        self.assertEqual(runtime.commits, [])

    def test_cancel_and_invalid_prediction(self):
        runtime = FakeRuntime([])
        with self.assertRaises(InterruptedError):
            resolve_block(runtime, width=2, mask_token=99, threshold=.95, should_stop=lambda: True)
        with self.assertRaises(ValueError):
            resolve_block(FakeRuntime([[Prediction(99, 1)]]), width=1, mask_token=99, threshold=.95)


class CoresetTests(unittest.TestCase):
    def test_stable_ties_and_mask(self):
        rows = [[2., 2., 0.], [0., 0., 100.]]
        self.assertEqual(select_expert_coreset(rows, [True, False], top_k=2, top_p=.5), (0,))
        self.assertEqual(select_expert_coreset(rows, [True, False], top_k=2, top_p=1), (0, 1))

    def test_invalid_values(self):
        with self.assertRaises(ValueError):
            select_expert_coreset([[float('nan'), 0]], [True], top_k=1, top_p=.9)
        with self.assertRaises(ValueError):
            select_expert_coreset([[0, 1]], [False], top_k=1, top_p=.9)


if __name__ == '__main__':
    unittest.main()
