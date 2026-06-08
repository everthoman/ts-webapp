"""
Multi-step (linear route) Thompson Sampling.

``RouteSampler`` generalises :class:`thompson_sampling.ThompsonSampler` from a
single reaction to an ordered sequence of reactions that build one final
product:

    step 0:  R0(reagent_a, reagent_b, ...)        -> intermediate_0
    step 1:  R1(intermediate_0, reagent_c)        -> intermediate_1
    step k:  Rk(intermediate_{k-1}, reagent_...)  -> intermediate_k
    ...
    final product = intermediate_last   (this is what gets scored)

Only the *final* product is passed to the evaluator, matching the requested
behaviour ("reactions applied sequentially and TS only applied to final
products").

The reagent components that Thompson Sampling samples over are the flat list of
"new reagent" inputs across all steps, in route order. The running intermediate
is threaded automatically and is never a sampled component. Everything else
(warm-up, search, the disallow tracker, the reagent priors) is inherited from
``ThompsonSampler`` unchanged. With a single step this reduces exactly to the
original single-reaction behaviour.
"""

from typing import List, Tuple

from rdkit import Chem
from rdkit.Chem import AllChem

from thompson_sampling import ThompsonSampler


class RouteSampler(ThompsonSampler):
    def __init__(self, mode="maximize", log_filename=None):
        super().__init__(mode=mode, log_filename=log_filename)
        # List of (compiled_reaction, num_new_reagents) in route order.
        self.route_steps: List[Tuple[AllChem.ChemicalReaction, int]] = []

    def set_route(self, steps: List[Tuple[str, int]]) -> None:
        """
        Define the reaction sequence.

        :param steps: list of ``(reaction_smarts, num_new_reagents)`` tuples in
            route order. ``num_new_reagents`` is the number of sampled reagent
            components the step consumes *in addition* to the running
            intermediate. The first step takes no intermediate, so the sum of
            ``num_new_reagents`` across all steps must equal the number of
            reagent components (``len(self.reagent_lists)``).
        """
        self.route_steps = [
            (AllChem.ReactionFromSmarts(smarts), int(n_new)) for smarts, n_new in steps
        ]

    def set_reaction(self, rxn_smarts):
        """Convenience: a single-step route equivalent to the base class."""
        self.set_route([(rxn_smarts, len(self.reagent_lists) or 1)])

    def _expected_reagent_count(self) -> int:
        return sum(n for _rxn, n in self.route_steps)

    def _build_product(self, choice_list: List[int]):
        """
        Build the final product by running the reaction sequence.

        Overrides the single-reaction base method so that all of the base
        sampler's machinery (sequential ``evaluate`` and parallel
        ``evaluate_batch``, warm-up and search) drives the multi-step route
        unchanged. Pure / no shared state, so it is safe to call from worker
        threads.

        :param choice_list: list of reagent indices, one per reagent component,
            ordered to match the flat ``reagent_lists`` (route order).
        :return: ``(product_mol_or_None, smiles, product_name, selected_reagents)``.
        """
        selected_reagents = [
            self.reagent_lists[idx][choice] for idx, choice in enumerate(choice_list)
        ]
        product_name = "_".join(r.reagent_name for r in selected_reagents)
        try:
            cursor = 0
            intermediate = None
            for rxn, n_new in self.route_steps:
                reactants = []
                if intermediate is not None:
                    reactants.append(intermediate)
                for _ in range(n_new):
                    reactants.append(selected_reagents[cursor].mol)
                    cursor += 1
                products = rxn.RunReactants(reactants)
                if not products:
                    return None, "FAIL", product_name, selected_reagents
                intermediate = products[0][0]  # Tuple[Tuple[Mol]]
                Chem.SanitizeMol(intermediate)
            product_smiles = Chem.MolToSmiles(intermediate)
        except Exception:
            # Any RDKit failure in the route -> treat as a failed product (NaN).
            return None, "FAIL", product_name, selected_reagents
        return intermediate, product_smiles, product_name, selected_reagents
