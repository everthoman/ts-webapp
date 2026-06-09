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
        # Stamp the reagent-combo name so evaluators can label products (e.g. the
        # live gallery reads it back); harmless for evaluators that ignore it.
        try:
            prod_mol.SetProp("_Name", product_name)
        except Exception:
            pass
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
        results, built = self._score_choice_lists(choice_lists)
        for (_smiles, _name, score, _reason), (_pm, _smi, _nm, selected_reagents) in zip(results, built):
            self._record(selected_reagents, score)
        return results

    def _score_choice_lists(self, choice_lists: List[List[int]]):
        """Build + dock a batch, ``self.concurrency`` at a time.

        Returns ``(results, built)`` where ``results`` is one
        ``(smiles, name, score, reason)`` per input (in input order) and
        ``built`` is the raw ``_build_product`` output (kept so callers can
        record into reagents). This does *no* posterior bookkeeping itself, so
        both the standard-TS :meth:`evaluate_batch` and the RWS search can drive
        the same threaded docking path.
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

        results = [(built[i][1], built[i][2], scores[i], reasons[i]) for i in range(len(built))]
        return results, built

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

    def search(self, num_cycles=25, patience=None):
        """Run the search
        :param: num_cycles: number of search iterations
        :param: patience: if set, stop early once the docked best score has not
            improved for this many consecutive docks (plateau auto-stop)
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
        # Measure the plateau window over search docks only, not warm-up.
        if patience and hasattr(self.evaluator, "reset_plateau"):
            self.evaluator.reset_plateau()
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
            if patience and getattr(self.evaluator, "docks_since_best", 0) >= patience:
                self.logger.info(
                    f"Auto-stop: no score improvement in {patience} docks (plateau) "
                    f"after {done} search iterations.")
                break
        pbar.close()
        return out_list

    # -- Roulette Wheel Selection (RWS) -------------------------------------
    # An alternative to the argmax TS search above, after Zhao et al., "Enhanced
    # Thompson sampling by roulette wheel selection for screening ultralarge
    # combinatorial libraries", J. Cheminform. 17:154 (2025), matching the
    # reference implementation at github.com/PatWalters/TS_2025. Two differences
    # from plain TS: (1) reagents are drawn by *roulette wheel* on Boltzmann
    # weights of their sampled scores (not argmax), with a per-component
    # temperature thermal-cycled to trade greediness for diversity; and (2) the
    # posterior mean is a Boltzmann-weighted average (Eq. 3), so a strong reagent
    # is not dragged down by its many weak random partners. RWS matches TS on
    # 2-component reactions and beats it on 3+-component ones — the multi-step
    # route case here. Posterior state is kept in the sampler as per-component
    # arrays so the standard-TS Reagent path is untouched. Scores are handled in
    # "maximize space" via ``scaling`` (+1 maximize, -1 minimize), matching the
    # reference.

    def _rws_scaling_factor(self) -> float:
        return 1.0 if self._mode.startswith("maximize") else -1.0

    def _rws_single_update(self, comb, value: float) -> None:
        """Boltzmann-weighted posterior update for one scored product (Eq. 3)."""
        for comp_i, reagent_j in enumerate(comb):
            prior_var = self._rws_std[comp_i][reagent_j] ** 2
            denominator = prior_var + self._rws_var_known
            w = math.exp(value / self._rws_std_known)
            self._rws_sumw[comp_i][reagent_j] += w
            sw = self._rws_sumw[comp_i][reagent_j]
            mu = self._rws_mu[comp_i][reagent_j]
            self._rws_mu[comp_i][reagent_j] = mu + (w / sw) * (value - mu)
            self._rws_std[comp_i][reagent_j] = np.sqrt(prior_var * self._rws_var_known / denominator)

    def warm_up_rws(self, num_warmup_trials=5):
        """RWS warm-up: dock a balanced set of random products (each reagent used
        >= ``num_warmup_trials`` times), then seed each reagent's Boltzmann-weighted
        posterior from the population prior plus its own warm-up scores.

        Returns the finite ``[score, smiles, name]`` rows (for the results CSV /
        gallery); returns ``[]`` if nothing scored so the caller can report it.
        """
        scaling = self._rws_scaling_factor()
        reagent_count_list = [len(x) for x in self.reagent_lists]
        rmax = max(reagent_count_list)

        # Balanced warm-up matrix: shuffle each component and tile shorter ones up
        # to rmax, so every reagent is paired at least num_warmup_trials times.
        pairs: List[List[int]] = []
        for _ in range(num_warmup_trials):
            matrix = []
            for nr in reagent_count_list:
                idx_r = list(range(nr))
                random.shuffle(idx_r)
                if nr < rmax:
                    matrix.append(idx_r * (rmax // nr) + idx_r[: rmax % nr])
                else:
                    matrix.append(idx_r)
            pairs.extend(np.array(matrix).transpose().tolist())

        warmup_results = []  # finite [score, smiles, name] (real, unscaled scores)
        batches = [[[] for _ in range(nr)] for nr in reagent_count_list]  # scaled scores per reagent
        for chunk in tqdm(list(self._chunks(pairs, self._batch_size())),
                          desc="RWS warmup", disable=self.hide_progress):
            results, _built = self._score_choice_lists(chunk)
            for choice_list, (smiles, name, score, _reason) in zip(chunk, results):
                if np.isfinite(score):
                    warmup_results.append([score, smiles, name])
                    for comp_i, reagent_j in enumerate(choice_list):
                        batches[comp_i][reagent_j].append(score * scaling)

        self.num_warmup = len(pairs)
        if not warmup_results:
            return []

        warmup_scores = [w[0] for w in warmup_results]
        prior_mean = float(np.mean(warmup_scores)) * scaling
        prior_std = float(np.std(warmup_scores))
        self._warmup_std = prior_std
        self.logger.info(
            f"RWS warmup score stats: cnt={len(warmup_scores)}, "
            f"mean={np.mean(warmup_scores):0.4f}, std={prior_std:0.4f}, "
            f"min={np.min(warmup_scores):0.4f}, max={np.max(warmup_scores):0.4f}")

        # Seed posteriors from the population prior, then fold in each reagent's
        # own warm-up scores via the Boltzmann-weighted batch update.
        self._rws_var_known = prior_std ** 2
        self._rws_std_known = prior_std if prior_std > 0 else 1.0
        self._rws_mu, self._rws_std, self._rws_sumw = [], [], []
        for comp_i, nr in enumerate(reagent_count_list):
            mu = np.full(nr, prior_mean, dtype=float)
            std = np.full(nr, prior_std, dtype=float)
            sumw = np.exp(mu / self._rws_std_known)
            for j in range(nr):
                sb = batches[comp_i][j]
                if sb:
                    sb_arr = np.array(sb)
                    prior_var = std[j] ** 2
                    denominator = len(sb) * prior_var + self._rws_var_known
                    w_batch = np.exp(sb_arr / self._rws_std_known)
                    mean_batch = np.average(sb_arr, weights=w_batch)
                    w_sum = float(np.sum(w_batch))
                    sumw[j] += w_sum
                    mu[j] = mu[j] + (w_sum / sumw[j]) * (mean_batch - mu[j])
                    std[j] = np.sqrt(prior_var * self._rws_var_known / denominator)
            self._rws_mu.append(mu)
            self._rws_std.append(std)
            self._rws_sumw.append(sumw)
        return warmup_results

    def search_rws(self, num_targets, min_cpds_per_core=50, stop=6000, patience=None):
        """RWS search with thermal cycling.

        Each cycle samples ``num_per_cycle`` reagents per component by roulette
        wheel on Boltzmann weights of their sampled scores; one rotating component
        is "heated" (temperature ``alpha``) while the rest are "cooled"
        (``beta``). Temperatures rise when too few new (unique) products are found,
        shifting from greedy to diverse. Unique products are docked in batches and
        their reagents' posteriors updated.

        ``num_targets`` is an absolute budget: the number of unique products to
        dock in the search phase (analogous to the standard-TS ``num_cycles``),
        so a TS vs RWS run on the same budget is apples-to-apples. Capped at the
        library size; also stops early after ``stop`` consecutive resamples once
        the reachable space is effectively exhausted.
        """
        scaling = self._rws_scaling_factor()
        rng = np.random.default_rng(self.seed)
        n_component = len(self.reagent_lists)
        nsearch = min(int(num_targets), max(1, int(self.num_prods)))
        if nsearch <= 0:
            return []

        num_per_cycle = 100
        se_threshold = num_per_cycle * 0.1  # heat up when fewer than this many are new
        min_cpds_per_batch = max(1, self.concurrency) * int(min_cpds_per_core)
        alpha = beta = 0.1
        idx_c = 0
        uniq = {}
        out_list = []
        pairs_u: List[List[int]] = []
        n_resample = 0
        count = 0

        if patience and hasattr(self.evaluator, "reset_plateau"):
            self.evaluator.reset_plateau()
        pbar = tqdm(total=nsearch, desc="RWS search", disable=self.hide_progress)
        try:
            while len(uniq) < nsearch:
                matrix = []
                for ii in range(n_component):
                    mu = self._rws_mu[ii]
                    std = self._rws_std[ii]
                    rg_score = rng.normal(size=len(mu)) * std + mu
                    spread = np.std(rg_score)
                    if spread == 0:
                        spread = 1.0
                    temp = alpha if ii == idx_c else beta
                    w = np.exp((rg_score - np.mean(rg_score)) / spread / temp)
                    matrix.append(rng.choice(len(mu), num_per_cycle, p=w / np.sum(w)))
                idx_c = (idx_c + 1) % n_component
                pairs = np.array(matrix).transpose()

                n_uniq = 0
                for comb in pairs:
                    key = "_".join(str(r) for r in comb)
                    if key not in uniq:
                        pairs_u.append(list(comb))
                        uniq[key] = None
                        n_resample = 0
                        n_uniq += 1
                    else:
                        n_resample += 1
                if n_resample >= stop:
                    self.logger.info(f"RWS stop: {n_resample} consecutive resamples (library effectively exhausted)")
                    break

                if len(uniq) < nsearch:
                    # Adaptive temperature: heat up when sampling efficiency drops.
                    if n_uniq < se_threshold:
                        alpha += 0.01
                        if n_uniq == 0:
                            beta += 0.001
                    # Defer docking until a full parallel batch has accumulated.
                    if len(pairs_u) < min_cpds_per_batch:
                        continue

                results, _built = self._score_choice_lists(pairs_u)
                for (smiles, name, score, _reason), comb in zip(results, pairs_u):
                    if np.isfinite(score):
                        out_list.append([score, smiles, name])
                        self._rws_single_update(comb, score * scaling)
                pbar.update(min(len(pairs_u), nsearch - pbar.n))
                pairs_u = []
                count += 1
                if patience and getattr(self.evaluator, "docks_since_best", 0) >= patience:
                    self.logger.info(
                        f"Auto-stop: no score improvement in {patience} docks (plateau) "
                        f"after {len(uniq)} unique products.")
                    break
                if count % 100 == 0 and out_list:
                    top_score, top_smiles, top_name = (max(out_list) if scaling > 0 else min(out_list))
                    self.logger.info(f"RWS iteration {count}: best {top_score:.3f} {top_smiles} {top_name}")
        finally:
            pbar.close()
        return out_list
