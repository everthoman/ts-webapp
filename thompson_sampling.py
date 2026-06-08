import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Tuple

import functools
import math
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm.auto import tqdm

from disallow_tracker import DisallowTracker
from reagent import Reagent
from ts_logger import get_logger
from ts_utils import read_reagents
from evaluators import DBEvaluator



class ThompsonSampler:
    def __init__(self, mode="maximize", log_filename: Optional[str] = None):
        """
        Basic init
        :param mode: maximize or minimize
        :param log_filename: Optional filename to write logging to. If None, logging will be output to stdout
        """
        # A list of lists of Reagents. Each component in the reaction will have one list of Reagents in this list
        self.reagent_lists: List[List[Reagent]] = []
        self.reaction = None
        self.evaluator = None
        self.num_prods = 0
        self.logger = get_logger(__name__, filename=log_filename)
        self._disallow_tracker = None
        self.hide_progress = False
        # Number of products to score in parallel. 1 == the original sequential
        # behaviour. The evaluator must be thread-safe (GninaEvaluator is).
        self.concurrency = 1
        # Seed for the search RNG (None == nondeterministic). warm-up uses the
        # global numpy/random state, which the caller seeds separately.
        self.seed = None
        self._mode = mode
        if self._mode == "maximize":
            self.pick_function = np.nanargmax
            self._top_func = max
        elif self._mode == "minimize":
            self.pick_function = np.nanargmin
            self._top_func = min
        elif self._mode == "maximize_boltzmann":
            # See documentation for _boltzmann_reweighted_pick
            self.pick_function = functools.partial(self._boltzmann_reweighted_pick)
            self._top_func = max
        elif self._mode == "minimize_boltzmann":
            # See documentation for _boltzmann_reweighted_pick
            self.pick_function = functools.partial(self._boltzmann_reweighted_pick)
            self._top_func = min
        else:
            raise ValueError(f"{mode} is not a supported argument")
        self._warmup_std = None

    def _boltzmann_reweighted_pick(self, scores: np.ndarray):
        """Rather than choosing the top sampled score, use a reweighted probability.

        Zhao, H., Nittinger, E. & Tyrchan, C. Enhanced Thompson Sampling by Roulette
        Wheel Selection for Screening Ultra-Large Combinatorial Libraries.
        bioRxiv 2024.05.16.594622 (2024) doi:10.1101/2024.05.16.594622
        suggested several modifications to the Thompson Sampling procedure.
        This method implements one of those, namely a Boltzmann style probability distribution
        from the sampled values. The reagent is chosen based on that distribution rather than
        simply the max sample.
        """
        if self._mode == "minimize_boltzmann":
            scores = -scores
        exp_terms = np.exp(scores / self._warmup_std)
        probs = exp_terms / np.nansum(exp_terms)
        probs[np.isnan(probs)] = 0.0
        return np.random.choice(probs.shape[0], p=probs)

    def set_hide_progress(self, hide_progress: bool) -> None:
        """
        Hide the progress bars
        :param hide_progress: set to True to hide the progress baars
        """
        self.hide_progress = hide_progress

    def set_concurrency(self, concurrency: int) -> None:
        """
        Number of products to build + dock in parallel (>=1).

        Docking is the rate-limiting step and each dock is independent, so the
        warm-up and search phases dock a batch of ``concurrency`` molecules at
        once. Scores are still recorded into the reagents single-threaded after
        each batch, so the Thompson Sampling bookkeeping is unchanged; only the
        wall-clock time drops. Requires a thread-safe evaluator.
        """
        self.concurrency = max(1, int(concurrency))

    def set_seed(self, seed: Optional[int]) -> None:
        """Seed for reproducible runs. ``None`` keeps the search nondeterministic."""
        self.seed = None if seed is None else int(seed)

    def read_reagents(self, reagent_file_list, num_to_select: Optional[int] = None):
        """
        Reads the reagents from reagent_file_list
        :param reagent_file_list: List of reagent filepaths
        :param num_to_select: Max number of reagents to select from the reagents file (for dev purposes only)
        :return: None
        """
        self.reagent_lists = read_reagents(reagent_file_list, num_to_select)
        self.num_prods = math.prod([len(x) for x in self.reagent_lists])
        self.logger.info(f"{self.num_prods:.2e} possible products")
        self._disallow_tracker = DisallowTracker([len(x) for x in self.reagent_lists])

    def get_num_prods(self) -> int:
        """
        Get the total number of possible products
        :return: num_prods
        """
        return self.num_prods

    def set_evaluator(self, evaluator):
        """
        Define the evaluator
        :param evaluator: evaluator class, must define an evaluate method that takes an RDKit molecule
        """
        self.evaluator = evaluator

    def set_reaction(self, rxn_smarts):
        """
        Define the reaction
        :param rxn_smarts: reaction SMARTS
        """
        self.reaction = AllChem.ReactionFromSmarts(rxn_smarts)

    def _build_product(self, choice_list: List[int]):
        """Build the product molecule for a set of reagent choices.

        :return: ``(product_mol_or_None, smiles, product_name, selected_reagents)``.
            ``product_mol`` is ``None`` (and smiles ``"FAIL"``) when the reaction
            does not fire or the product cannot be sanitised. Pure / no shared
            state, so it is safe to call this from worker threads.
        """
        selected_reagents = [
            self.reagent_lists[idx][choice] for idx, choice in enumerate(choice_list)
        ]
        product_name = "_".join(r.reagent_name for r in selected_reagents)
        try:
            prod = self.reaction.RunReactants([r.mol for r in selected_reagents])
            if not prod:
                return None, "FAIL", product_name, selected_reagents
            prod_mol = prod[0][0]  # RunReactants returns Tuple[Tuple[Mol]]
            Chem.SanitizeMol(prod_mol)
            product_smiles = Chem.MolToSmiles(prod_mol)
        except Exception:
            return None, "FAIL", product_name, selected_reagents
        return prod_mol, product_smiles, product_name, selected_reagents

    def _score_product(self, prod_mol, product_name: str) -> float:
        """Score one product. This is the slow (docking) step run in parallel."""
        return self._score_product_detailed(prod_mol, product_name)[0]

    def _score_product_detailed(self, prod_mol, product_name: str):
        """Score one product, returning ``(score, reason)``.

        ``reason`` is ``None`` for a real score, ``"filtered"`` if the evaluator
        rejected it pre-dock, or ``"fail"`` for a prep/dock failure. Evaluators
        that don't classify failures (no ``evaluate_detailed``) report ``"fail"``
        for any non-finite score.
        """
        if isinstance(self.evaluator, DBEvaluator):
            score = float(self.evaluator.evaluate(product_name))
            return score, (None if np.isfinite(score) else "fail")
        detailed = getattr(self.evaluator, "evaluate_detailed", None)
        if detailed is not None:
            return detailed(prod_mol)
        score = self.evaluator.evaluate(prod_mol)
        return score, (None if np.isfinite(score) else "fail")

    def _record(self, selected_reagents, score: float) -> None:
        """Record a finite score against each contributing reagent."""
        if np.isfinite(score):
            for reagent in selected_reagents:
                reagent.add_score(score)

    def evaluate(self, choice_list: List[int]) -> Tuple[str, str, float]:
        """Evaluate a set of reagents
        :param choice_list: list of reagent ids
        :return: smiles for the reaction product, score for the reaction product
        """
        prod_mol, product_smiles, product_name, selected_reagents = self._build_product(choice_list)
        res = np.nan
        if prod_mol is not None:
            res = self._score_product(prod_mol, product_name)
            self._record(selected_reagents, res)
        return product_smiles, product_name, res

    def evaluate_batch(self, choice_lists: List[List[int]]) -> List[Tuple[str, str, float, Optional[str]]]:
        """Evaluate several reagent sets, docking up to ``self.concurrency`` at once.

        Products are built (cheap) on the calling thread, docked in parallel,
        then scores are recorded into the reagents single-threaded so the TS
        bookkeeping is identical to the sequential path. Returns one
        ``(smiles, name, score, reason)`` per input, in input order. ``reason``
        is ``None`` for a real score, ``"reaction"`` if the reaction did not fire,
        ``"filtered"`` if a hard filter rejected it, or ``"fail"`` for a prep/dock
        failure.
        """
        built = [self._build_product(cl) for cl in choice_lists]
        scores: List[float] = [np.nan] * len(built)
        # Products that did not build = reaction never fired.
        reasons: List[Optional[str]] = ["reaction" if b[0] is None else None for b in built]
        dock_idx = [i for i, b in enumerate(built) if b[0] is not None]

        if self.concurrency > 1 and len(dock_idx) > 1:
            pending_exc = None
            with ThreadPoolExecutor(max_workers=self.concurrency) as ex:
                futs = {ex.submit(self._score_product_detailed, built[i][0], built[i][2]): i for i in dock_idx}
                for fut in as_completed(futs):
                    try:
                        scores[futs[fut]], reasons[futs[fut]] = fut.result()
                    except Exception as e:  # e.g. DockingCancelled
                        pending_exc = e
                        for f in futs:
                            f.cancel()  # drop not-yet-started docks
                        break
            if pending_exc is not None:
                raise pending_exc
        else:
            for i in dock_idx:
                scores[i], reasons[i] = self._score_product_detailed(built[i][0], built[i][2])

        out = []
        for i, (_pm, smiles, name, selected_reagents) in enumerate(built):
            self._record(selected_reagents, scores[i])
            out.append((smiles, name, scores[i], reasons[i]))
        return out

    @staticmethod
    def _chunks(seq, n):
        for i in range(0, len(seq), n):
            yield seq[i:i + n]

    def _batch_size(self) -> int:
        """How many products to build per chunk (bounds peak memory)."""
        return self.concurrency * 8 if self.concurrency > 1 else 1

    def warm_up(self, num_warmup_trials=3):
        """Warm-up phase, each reagent is sampled with num_warmup_trials random partners
        :param num_warmup_trials: number of times to sample each reagent
        """
        # get the list of reagent indices
        idx_list = list(range(0, len(self.reagent_lists)))
        # get the number of reagents for each component in the reaction
        reagent_count_list = [len(x) for x in self.reagent_lists]

        # Phase 1: enumerate every warm-up reagent combination. The disallow
        # tracking and random partner selection are cheap CPU work and stay
        # sequential (and deterministic w.r.t. the RNG) — only docking, in
        # phase 2, is parallelised.
        combos: List[List[int]] = []
        for i in idx_list:
            partner_list = [x for x in idx_list if x != i]
            # The number of reagents for this component
            current_max = reagent_count_list[i]
            # For each reagent...
            for j in range(0, current_max):
                # For each warmup trial...
                for k in range(0, num_warmup_trials):
                    current_list = [DisallowTracker.Empty] * len(idx_list)
                    current_list[i] = DisallowTracker.To_Fill
                    disallow_mask = self._disallow_tracker.get_disallowed_selection_mask(current_list)
                    if j not in disallow_mask:
                        ## ok we can select this reagent
                        current_list[i] = j
                        # Randomly select reagents for each additional component of the reaction
                        for p in partner_list:
                            # tell the disallow tracker which site we are filling
                            current_list[p] = DisallowTracker.To_Fill
                            # get the new disallow mask
                            disallow_mask = self._disallow_tracker.get_disallowed_selection_mask(current_list)
                            selection_scores = np.random.uniform(size=reagent_count_list[p])
                            # null out the disallowed ones
                            selection_scores[list(disallow_mask)] = np.nan
                            # and select a random one
                            current_list[p] = np.nanargmax(selection_scores).item(0)
                        self._disallow_tracker.update(current_list)
                        combos.append(list(current_list))

        # Phase 2: dock the combinations, self.concurrency at a time. Track, per
        # reagent, how many of its warm-up products failed for a *genuine* reason
        # (reaction didn't fire / prep / dock) as opposed to being filtered out
        # on a property/structural alert — filtered pairings must not retire a
        # reagent (a good building block can have out-of-range random partners).
        warmup_results = []
        realfail_counts = {}  # (component_idx, reagent_idx) -> count of genuine failures
        for chunk in tqdm(list(self._chunks(combos, self._batch_size())),
                          desc="Warmup", disable=self.hide_progress):
            for choice_list, (product_smiles, product_name, score, reason) in zip(chunk, self.evaluate_batch(chunk)):
                if np.isfinite(score):
                    warmup_results.append([score, product_smiles, product_name])
                elif reason in ("reaction", "fail"):
                    for comp_i, reagent_j in enumerate(choice_list):
                        realfail_counts[(comp_i, reagent_j)] = realfail_counts.get((comp_i, reagent_j), 0) + 1

        warmup_scores = [ws[0] for ws in warmup_results]
        self.logger.info(
            f"warmup score stats: "
            f"cnt={len(warmup_scores)}, "
            f"mean={np.mean(warmup_scores):0.4f}, "
            f"std={np.std(warmup_scores):0.4f}, "
            f"min={np.min(warmup_scores):0.4f}, "
            f"max={np.max(warmup_scores):0.4f}")
        # initialize each reagent
        prior_mean = np.mean(warmup_scores)
        prior_std = np.std(warmup_scores)
        self._warmup_std = prior_std
        kept_filtered = 0
        for i in range(0, len(self.reagent_lists)):
            for j in range(0, len(self.reagent_lists[i])):
                reagent = self.reagent_lists[i][j]
                try:
                    reagent.init_given_prior(prior_mean=prior_mean, prior_std=prior_std)
                except ValueError:
                    # No successful warm-up score for this reagent.
                    if realfail_counts.get((i, j), 0) == 0:
                        # Every attempt was merely filtered (or it was never
                        # sampled) — keep it with the population prior so tight
                        # filters don't silently eliminate a good building block.
                        reagent.init_prior_no_obs(prior_mean=prior_mean, prior_std=prior_std)
                        kept_filtered += 1
                    else:
                        # Genuine reaction/prep/dock failures — retire it.
                        self.logger.info(f"Skipping reagent {reagent.reagent_name} because its warm-up products failed to react or dock")
                        self._disallow_tracker.retire_one_synthon(i, j)
        if kept_filtered:
            self.logger.info(
                f"Kept {kept_filtered} reagent(s) with the population prior "
                f"(all warm-up products were filtered, not undockable)")
        self.logger.info(f"Top score found during warmup: {max(warmup_scores):.3f}")
        return warmup_results

    def _sample_one(self, rng) -> List[int]:
        """Draw one reagent selection from the current TS posteriors.

        Raises ``ValueError`` (from the nanargmax/min pick) once the disallow
        tracker has exhausted the library, which the caller treats as "stop".
        """
        selected_reagents = [DisallowTracker.Empty] * len(self.reagent_lists)
        for cycle_id in random.sample(range(0, len(self.reagent_lists)), len(self.reagent_lists)):
            reagent_list = self.reagent_lists[cycle_id]
            selected_reagents[cycle_id] = DisallowTracker.To_Fill
            disallow_mask = self._disallow_tracker.get_disallowed_selection_mask(selected_reagents)
            stds = np.array([r.current_std for r in reagent_list])
            mu = np.array([r.current_mean for r in reagent_list])
            choice_row = rng.normal(size=len(reagent_list)) * stds + mu
            if disallow_mask:
                choice_row[np.array(list(disallow_mask))] = np.nan
            selected_reagents[cycle_id] = self.pick_function(choice_row)
        self._disallow_tracker.update(selected_reagents)
        return selected_reagents

    def search(self, num_cycles=25):
        """Run the search
        :param: num_cycles: number of search iterations
        :return: a list of SMILES and scores

        Iterations are processed in batches of ``self.concurrency`` so docking
        runs in parallel. Within a batch the selections are drawn from the same
        (not-yet-updated) posteriors — the standard batched Thompson Sampling
        trade-off — which is exact when concurrency == 1.
        """
        out_list = []
        rng = np.random.default_rng(self.seed)
        batch_size = max(1, self.concurrency)
        done = 0
        last_log = 0
        exhausted = False
        pbar = tqdm(total=num_cycles, desc="Cycle", disable=self.hide_progress)
        while done < num_cycles and not exhausted:
            selections = []
            for _ in range(min(batch_size, num_cycles - done)):
                try:
                    selections.append(self._sample_one(rng))
                except ValueError:
                    exhausted = True
                    break
            if not selections:
                break
            for smiles, name, score, _reason in self.evaluate_batch(selections):
                if np.isfinite(score):
                    out_list.append([score, smiles, name])
            done += len(selections)
            pbar.update(len(selections))
            if out_list and done - last_log >= 100:
                top_score, top_smiles, top_name = self._top_func(out_list)
                self.logger.info(f"Iteration: {done} max score: {top_score:2f} smiles: {top_smiles} {top_name}")
                last_log = done
        pbar.close()
        return out_list
