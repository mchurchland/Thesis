# -*- coding: utf-8 -*-
"""pure_nomad_generic.py
====================================================
A **general‑purpose evolutionary optimiser** that couples global search
(crossover + mutation) with local refinement via the NOMAD derivative‑free
solver (through *PyNomad*). It is *not* tied to any specific domain – the user
supplies an ``objective_fn`` that maps a NumPy 1‑D parameter vector → scalar
fitness.  Higher fitness means *better*.  The optimiser handles everything
else.

Example
-------
```python
import numpy as np
from EANOMAD import EA

def sphere(x: np.ndarray) -> float:
    return -np.sum(x ** 2)         # maximise ⇒ minimise (–sphere)

opt = EA(
    population_size=64,
    dimension=30,
    objective_fn=sphere,
    subset_size=10,
    bounds=0.2,
    max_bb_eval=250,
)

best_x, best_fit = opt.run(generations=200)
print(best_fit, best_x)
```

Dependencies: ``numpy``, ``PyNomad>=0.9`` (optional ``ray`` for parallel NOMAD
calls; falls back to serial execution if not available).
"""
from __future__ import annotations

import math
import numpy as np
from tqdm import tqdm                      
import numpy.typing as npt
from typing import Callable, Tuple, Optional, List
import warnings
from numpy.random import Generator

try:
    import ray  # type: ignore
    from ray.actor import ActorHandle
    _RAY_AVAILABLE = True
except ImportError:  # pragma: no cover
    _RAY_AVAILABLE = False

import PyNomad  # type: ignore

__all__ = [
    "EANOMAD",
]

# ─────────────────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────────────────

def _uniform_crossover(parents: npt.NDArray, probs: npt.NDArray, rng: Generator,crossover_prob: float) -> npt.NDArray:
    """Vectorised uniform crossover.

    Parameters
    ----------
    parents : (P, D) parent weight matrix
    probs   : (P,) selection probabilities ∝ fitness (must sum to 1)

    Returns
    -------
    offspring : (P, D)  new individuals (same count as parents)
    """
    P, D = parents.shape
    p1_idx = rng.choice(P, size=P, p=probs)
    p2_idx = rng.choice(P, size=P, p=probs)

    p1, p2 = parents[p1_idx], parents[p2_idx]
    mask = rng.random((P, D)) < crossover_prob  # 50‑50 gene swap
    return np.where(mask, p1, p2)

def _fitness_based_crossover(
    parents: npt.NDArray,
    fits: npt.NDArray,
    rng: Generator, 
    exponent: float = 1.0,

) -> npt.NDArray:
    """Fitness‑biased uniform crossover (vectorised).

    For every offspring we draw two parents *with replacement* using fitness‑
    proportional selection.  For each gene we then choose which parent's value
    to keep with probability:

        P(gene←parent₁) = (fit₁ / (fit₁ + fit₂)) ** exponent

    A value of ``exponent > 1`` increases the bias towards the fitter parent;
    ``exponent = 1`` reduces to the linear case; ``exponent < 1`` makes the
    bias milder.
    """
    P, D = parents.shape
    probs = fits / fits.sum()

    # choose parents for each offspring ------------------------------------------------
    p1_idx = rng.choice(P, size=P, p=probs)
    p2_idx = rng.choice(P, size=P, p=probs)

    p1, p2 = parents[p1_idx], parents[p2_idx]
    f1, f2 = fits[p1_idx], fits[p2_idx]

    # compute per‑offspring bias, then broadcast to all genes -------------------------
    bias = (f1 / (f1 + f2)) ** exponent               # (P,)
    mask = rng.random((P, D)) < bias[:, None]       # (P,D)
    return np.where(mask, p1, p2)

def _random_reset_mutation(pop: npt.NDArray, n_coords: int, low: float, high: float, rng: Generator) -> None:
    """In‑place random‑reset mutation on *n_coords* indices per individual."""
    N, D = pop.shape
    for i in range(N):
        idx = rng.choice(D, n_coords, replace=False)
        pop[i, idx] = rng.uniform(low, high, size=n_coords)


# ─────────────────────────────────────────────────────────────────────────────
# NOMAD wrapper (minimises –fitness)
# ─────────────────────────────────────────────────────────────────────────────
# inside EANOMAD.core
_INT32_MAX = 2_147_483_600          # just under 2³¹‑1

def _safe_float(x) -> float:
    """Return a finite Python float in the PyNomad‑safe range."""
    y = float(np.asarray(x).item())          # ndarray / DeviceArray → float
    if not math.isfinite(y):
        return math.copysign(_INT32_MAX, y)
    return max(-_INT32_MAX, min(_INT32_MAX, y))
@staticmethod
def _nomad_local_search(
    fitness_fn: Callable[[npt.NDArray], float],
    x0: npt.NDArray,                    # starting point (sub‑vector)
    full_x: npt.NDArray,                # full vector (will be copied)
    ind: npt.NDArray[np.intp],          # indices we optimise (1‑D)
    max_bb_eval: int,
    bounds: float| None = None,
    lb: list | None = None,
    ub: list | None = None,
) -> Tuple[npt.NDArray, float]:
    """Run NOMAD on the coordinate slice ``ind``.
    Returns the updated *full* vector and its fitness.
    """
    if lb is None:
        lb = (x0 - bounds).tolist()
    if ub is None:
        ub = (x0 + bounds).tolist()

    def obj(eval_point):
        # copy vector and apply slice
        vect = full_x.copy()
        #print(vect[ind],[eval_point.get_coord(k) for k in range(len(ind))])
        vect[ind] = [eval_point.get_coord(k) for k in range(len(ind))]
        
        # single sign flip: NOMAD minimizes → we maximize
        val = -fitness_fn(vect)

        # guarantee a built‑in float and clamp insane values
        if not math.isfinite(val):
            val = 1e308          # largest finite double
        eval_point.setBBO(str(val).encode('utf-8'))
        return _safe_float(val)


    def blackbox_block(eval_block):
        eval_state = []
        for index in range(eval_block.size()):
            eval_point = eval_block.get_x(index)
            eval_state.append(obj(eval_point))
        return eval_state

    opts = [
            'DISPLAY_DEGREE 0', 
            'DISPLAY_STATS BBE BLK_SIZE OBJ', 
            'BB_MAX_BLOCK_SIZE 4',
            f'MAX_BB_EVAL {max_bb_eval}'
    ]

    res = PyNomad.optimize(blackbox_block, x0.tolist(), lb, ub, opts)
    best_slice = np.asarray(res["x_best"], dtype=np.float64)
    if best_slice.size != ind.size:      # NOMAD didn’t return a point
        best_slice = x0                  # keep the original slice
        best_fit   = _safe_float(fitness_fn(full_x))  # evaluate it
        full_out = full_x.copy() ##defensive can be removed
    else:
        best_fit   = _safe_float(-res["f_best"])
        full_out = full_x.copy()
        full_out[ind] = best_slice
    return full_out, best_fit


if _RAY_AVAILABLE:
    # Ray remote wrapper -----------------------------------------------------
    @ray.remote(num_cpus=1)
    def _nomad_remote(fn, x0, full_x, ind, max_bb_eval,bounds=None,lb=None,ub=None):  # type: ignore[valid-type]
        return _nomad_local_search(fn, x0, full_x, ind, max_bb_eval,bounds,lb,ub)
    @ray.remote(num_cpus=1)
    def _evaluate_individual_remote(obj:Callable,ind):
        return obj(ind)

# ─────────────────────────────────────────────────────────────────────────────
# Main optimiser class
# ─────────────────────────────────────────────────────────────────────────────

class EANOMAD:
    """Composable evolutionary optimiser with NOMAD local search. Plz use RAY
    Parameters
    ----------
    optimizer_type : {'EA', 'rEA'}
        • **'EA'** – every generation runs NOMAD on *all* individuals  
        • **'rEA'** – sparsity‑aware NOMAD + plain fitness evaluations

    population_size : int
        Number of individuals in the population (μ).

    dimension : int, optional
        Length of the parameter vector. Ignored if *init_pop* or *init_vec* is supplied.

    objective_fn : Callable[[np.ndarray], float]
        Objective to **maximise**. Receives a 1‑D parameter array and returns a scalar fitness.

    subset_size : int, default 20
        Number of coordinates optimised by NOMAD for each selected individual
        (must be < 50 due to NOMAD’s internal limit).

    bounds : float, default 0.1
        Half‑width of the ±box constraints passed to NOMAD around the current point.

    max_bb_eval : int, default 200
        Black‑box evaluation budget per NOMAD call.

    n_elites : int, optional
        Top‑k individuals refined each generation.  
        Defaults to ``population_size // 2``.

    n_mutate_coords : int, default 0
        How many coordinates are replaced by a random value (*random‑reset mutation*)
        in each offspring.

    crossover_type : {'uniform', 'fitness'}, default 'uniform'
        • **'uniform'** – classic 50‑50 gene mask  
        • **'fitness'** – fitness‑biased per‑gene choice (see `crossover_exponent`).

    crossover_exponent : float, default 1.0
        Strength of the fitness bias when `crossover_type='fitness'`
        (1 → linear, >1 → sharper bias, <1 → milder).

    crossover_rate : float ∈ [0, 1), default 0.5
        Fraction of the next generation produced via crossover
        (the rest comes directly from the elite parents).

    init_pop : np.ndarray, shape (μ, D), optional
        Explicit initial population. Overrides *dimension*.

    init_vec : np.ndarray, shape (D,), optional
        Single vector copied μ times to form the initial population.
        Mutually exclusive with *init_pop*.

    low, high : float, default (-1.0, 1.0)
        Range for random initialisation and mutation.

    use_ray : bool, optional
        If **True** and Ray is installed, run NOMAD (and fitness calls in *hybrid* mode)
        in parallel. Defaults to *auto‑detect Ray*.

    seed : int, optional
        RNG seed for reproducibility.

    """

    def __init__(
        self,
        optimizer_type: str,
        population_size: int,
        *,
        dimension: int | None = None,
        objective_fn: Callable[[npt.NDArray], float],
        subset_size: int = 49,
        bounds: float | tuple[list,list] = 1.0,
        max_bb_eval: int = 200,
        n_mutate_coords: int = 0,
        crossover_type: str = "uniform",
        crossover_exponent: float = 1.0,
        crossover_rate: float = 0.5,
        steps_between_evo: int =1,
        init_pop: npt.NDArray | None = None,
        init_vec: npt.NDArray | None = None,
        low: float|None = None,
        high: float|None = None,
        use_ray: bool | None = None,
        seed: int | None = None,
    ) -> None:
        _valid_crossover = ['uniform','fitness']
        _valid_optimizer = ['EA','rEA']
        if (init_pop is None) and (init_vec is None) and dimension is None:
            raise ValueError("Either 'dimension' or 'init_pop' must be provided.")
        if init_pop and init_vec:
            raise ValueError("Only one of init_pop or init_vec should be given")
        
        if init_pop is not None:
            if init_pop.ndim < 2:
                raise ValueError("init_pop should be at least 2 dimensions"\
                                f"(given: {init_pop.ndim})") 
        if optimizer_type not in _valid_optimizer:
            raise ValueError(f"Invalid option provided for crossover_type: {optimizer_type}"\
                            f" valid options are {_valid_optimizer}")
        if optimizer_type == _valid_optimizer[1] and init_vec is None:
            raise ValueError(f"If you are using {_valid_optimizer[1]} you must pass in an init vector")
        if crossover_type not in _valid_crossover:
            raise ValueError(f"Invalid option provided for crossover_type: {crossover_type}"\
                             f" valid options are {_valid_crossover}")
        if crossover_rate < 0 or crossover_rate >= 1:
            raise ValueError("Crossover rate needs to be between 0 and 1"\
                             f"was given {crossover_rate}")
        if optimizer_type == "rEA" and n_mutate_coords <1:
            raise ValueError(f"When using rEA you must mutate cords, n_mutate_coords must be >0" \
                f"{n_mutate_coords}")
        if isinstance(bounds, tuple):
            lb, ub = bounds
            if len(lb) != len(ub):
                raise ValueError("Lower‑ and upper‑bound arrays must have the same length.")
            if any(l >= u for l, u in zip(lb, ub)):
                raise ValueError("Each lower bound must be strictly less than its upper bound.")

            self.lb, self.ub, self.bounds = lb, ub, None
        elif isinstance(bounds,float) or isinstance(bounds,int):
            self.bounds = bounds
            self.lb,self.ub = None,None
        else:
            raise ValueError("Bounds must be float, int or tuple of lists")


        self.D = (dimension if init_pop is None else init_pop.shape[1]) if init_vec is None else init_vec.shape[0]
        if isinstance(self.D,int):
            if subset_size > self.D:
                subset_size = min(49,self.D) ## 49 for nomads limit
                raise ValueError(f"subset_size should always be smaller than dimsensions we will automatically set it to the largest size possible ({subset_size})")
        else:
            raise ValueError("Something went wrong with setting dimension please ensure that init_pop\
                              has shape[1] or init_vec has shape[0] whichever you are using")
        if subset_size > 49:
            raise ValueError(f"Subset size must be less than 49 for nomad to work given: {subset_size}")
        
        if not seed:
            seed = np.random.randint(1, 100001)
        self.rng = np.random.default_rng(seed)

        self.mu = population_size
        # population ----------------------------------------------------
        if init_pop is not None:
            if init_pop.shape != (population_size, self.D):
                raise ValueError("init_pop must have shape (population_size, dimension)")
            self.pop: npt.NDArray = init_pop.copy()
            if low is None:
                self.low = np.min(init_pop)
                if n_mutate_coords > 0:
                    warnings.warn("low end for random mutations not supplied using\n"\
                                  "min value of init pop")
            else:
                self.low = low
            if high is None:
                self.high = np.max(init_pop)
                if n_mutate_coords > 0:
                    warnings.warn("low end for random mutations not supplied using\n"\
                                  "min value of init pop")
            else:
                self.high = high
        elif init_vec is not None:
            self.pop:npt.NDArray = np.array([init_vec for _ in range(self.mu)])
            if low is None:
                self.low = np.min(init_vec)
                if n_mutate_coords > 0:
                    warnings.warn("low end for random mutations not supplied using\n"\
                                  "min value of init vector")
            else:
                self.low = low
            if high is None:
                if n_mutate_coords > 0:
                    warnings.warn("high end for random mutations not supplied using\n"\
                                  "max value of init vector")
                self.high = np.max(init_vec)
            else:
                self.high = high
        else:
            if high is None:
                self.high = 1
            else:
                self.high = high
            if low is None:
                self.low = -1
            else:
                self.low = low
            self.pop = self.rng.uniform(self.low, self.high, size=(population_size, self.D))
        

        self.population_size = population_size
        self.optimizer_type  = optimizer_type
        self.crossover_type = crossover_type
        self.init_vec = init_vec
        self.crossover_exponent = crossover_exponent
        self.obj = objective_fn
        self.fitness_history: dict[str, np.ndarray] | None = None
        self._last_fits: np.ndarray | None = None
        self.subset_size = subset_size
        self.steps_between_evo = steps_between_evo
        self.max_bb_eval = max_bb_eval
        self.n_mutate_coords = n_mutate_coords
        self.crossover_rate = crossover_rate
        
        self.use_ray = _RAY_AVAILABLE if use_ray is None else use_ray and _RAY_AVAILABLE


        if self.use_ray:

            if not  ray.is_initialized():
                ray.init(ignore_reinit_error=True, log_to_driver=False)
            self.obj_actor: ActorHandle[ObjectiveActor] = ObjectiveActor.remote(self.obj)


        # internal bookkeeping -----------------------------------------
        self.generation = 1
        self._best_fit = -math.inf
        self._best_x: Optional[npt.NDArray] = None

    # ────────────────────────────────────────────────────────────────
    # Core loop
    # ────────────────────────────────────────────────────────────────

    def _evaluate_population(self) -> npt.NDArray[np.float64]:
        """Vectorised objective call; returns (μ,) fitness array."""
        return np.asarray([self.obj(ind) for ind in self.pop], dtype=np.float64)
    def _evaluate_individual(self,ind) -> np.float64:
        """Objective call; returns (μ,) fitness"""
        return self.obj(ind)
    



    def _select_parents(self, fits: npt.NDArray[np.float64]) -> Tuple[npt.NDArray, npt.NDArray]:
        idx = np.argsort(fits)[-self.mu // 2:]  # top‑half
        return self.pop[idx], fits[idx]

    def _make_offspring(self, parents: npt.NDArray, parent_fits: npt.NDArray,crossover_prob:float,crossover_expenent:float,crossover_type:str) -> npt.NDArray:
        if parent_fits.sum() ==0 and crossover_type=="fitness" and self.generation==1:
            warnings.warn("All parent have a fitness sum of 0, fitness based crossover\n"\
                          " does not work when this is the case (and will be skipped)\n" \
                          " for we will switch to uniform crossover")
        if parent_fits.sum() ==0:
            return parents
        if crossover_type =="uniform":
                probs = parent_fits / parent_fits.sum()
                offspring = _uniform_crossover(parents, probs,self.rng,crossover_prob)
        if crossover_type =="fitness":
                offspring = _fitness_based_crossover(parents, parent_fits,self.rng,crossover_expenent)
            
        _random_reset_mutation(offspring, self.n_mutate_coords, self.low, self.high,self.rng)
        return offspring

    def _NOMAD_search(self) -> List:
        tasks = []
        for idx in range(self.population_size):
            indiv = self.pop[idx]
            slice_idx = self.rng.choice(self.D, self.subset_size, replace=False)
            x0 = indiv[slice_idx].copy()
            lb_slice = [self.lb[i] for i in slice_idx] if self.lb is not None else None
            ub_slice = [self.ub[i] for i in slice_idx] if self.ub is not None else None
            if self.use_ray:
                tasks.append(
                    self.obj_actor.nomad.remote(x0, indiv, slice_idx, self.max_bb_eval, self.bounds, lb_slice, ub_slice))
            else:
                tasks.append(_nomad_local_search(self.obj, x0, indiv, slice_idx,self.max_bb_eval,
                                                 self.bounds,lb_slice,ub_slice))
            
        results = (ray.get(tasks) if self.use_ray else tasks)
        pop_fit = np.zeros(self.mu)
        for i,(x_new, fitness) in enumerate(results):
            self.pop[i] = x_new
            pop_fit[i] = fitness
        return pop_fit


    # ────────────────────────────────────────────────────────────────
    # Public API
    # ────────────────────────────────────────────────────────────────

    def step_pure(self) -> Tuple[float, npt.NDArray]:
        """Run one generation; return best fitness and best vector."""
        fits = self._NOMAD_search()
        self._last_fits = fits
        # book‑keep global best ---------------------------------------
        best_idx = np.argmax(fits)
        if fits[best_idx] > self._best_fit:
            self._best_fit = float(fits[best_idx])
            self._best_x   = self.pop[best_idx].copy()

        if self.generation % self.steps_between_evo==0:
            # produce next generation ------------------------------------
            parents, parent_fits = self._select_parents(fits)
            offspring = self._make_offspring(parents, parent_fits,self.crossover_rate,\
                                            self.crossover_exponent,self.crossover_type)

            stack:npt.NDArray = np.vstack([parents, offspring])

            if self.generation ==1 and stack.shape[0]<self.mu:
                warnings.warn(f"After crossover the population size({stack.shape[0]}).\n is less than specificied size ({self.mu}) may cause problems with population diversity", category=RuntimeWarning)
            elif self.generation ==1 and stack.shape[0]>self.mu:
                warnings.warn(f"After crossover the population size({stack.shape[0]}).\n exceeds specificied size ({self.mu}) will truncate crossover", category=RuntimeWarning)
            self.pop = stack[: self.mu]
        self.generation += 1
        return self._best_fit, self._best_x.copy() # type: ignore 

    
    def step_hybrid(self) -> Tuple[float, npt.NDArray]:
        """Hybrid GA + NOMAD generation (Ray‑parallelisable)."""

        record_masks: List[npt.NDArray[np.intp]] = []
        tasks = []           # Ray futures or local results
        idx_map = []         # original indices so we can put results back
        
        # -----------------------------------------------------------------
        # 1.  schedule local searches / fitness calls
        # -----------------------------------------------------------------
        for i, indiv in enumerate(self.pop):
            diff_idx = np.where(indiv != self.init_vec)[0]

            if 0 < diff_idx.size < 50 and not any(np.array_equal(diff_idx, m) for m in record_masks):
                # novel sparse mask → targeted NOMAD
                record_masks.append(diff_idx)
                x0 = indiv[diff_idx].copy()

                if self.use_ray:
                    tasks.append(self.obj_actor.nomad.remote(x0, indiv, diff_idx, self.max_bb_eval, self.bounds, self.lb, self.ub))
                else:
                    tasks.append(_nomad_local_search(
                        self.obj, x0, indiv, diff_idx, self.max_bb_eval,self.bounds, self.lb,self.ub
                    ))
            else:
                # plain fitness evaluation (no NOMAD)
                if self.use_ray:
                    tasks.append(self.obj_actor.eval.remote(indiv))
                else:
                    tasks.append(self._evaluate_individual(indiv))
            idx_map.append(i)

        # -----------------------------------------------------------------
        # 2.  collect results
        # -----------------------------------------------------------------
        results = ray.get(tasks) if self.use_ray else tasks
        fits = np.empty(self.mu, dtype=float)

        for tgt_idx, res in zip(idx_map, results):
            if isinstance(res, tuple):           # NOMAD returned (new_x, new_fit)
                new_x, new_fit = res
                self.pop[tgt_idx] = new_x
                fits[tgt_idx] = new_fit
            else:                                # plain fitness scalar
                fits[tgt_idx] = res
        self._last_fits = fits
        # -----------------------------------------------------------------
        # 3.  evolutionary cycle
        # -----------------------------------------------------------------
        best_idx = int(np.argmax(fits))
        if fits[best_idx] > self._best_fit:
                self._best_fit = float(fits[best_idx])
                self._best_x = self.pop[best_idx].copy()
        if self.generation % self.steps_between_evo==0:


            parents, parent_fits = self._select_parents(fits)
            offspring = self._make_offspring(
                parents, parent_fits,
                self.crossover_rate, self.crossover_exponent, self.crossover_type
            )
            stack:npt.NDArray = np.vstack([parents, offspring])
            if self.generation ==0 and stack.shape[0]<self.mu:
                warnings.warn(f"After crossover the population size({stack.shape[0]}).\n is less than specificied size ({self.mu}). This may cause problems with population diversity", category=RuntimeWarning)
            elif self.generation ==0 and stack.shape[0]>self.mu:
                warnings.warn(f"After crossover the population size({stack.shape[0]}).\n exceeds specificied size ({self.mu}). This truncates generated offspring", category=RuntimeWarning)
            self.pop = stack[: self.mu]
        self.generation += 1
        return self._best_fit, self._best_x.copy() # type: ignore 

    def run(self, generations: int) -> Tuple[npt.NDArray, float]:
        """Run optimisation for `generations` steps. Return best_x, best_fit."""
        step_fn:Callable = self.step_pure if self.optimizer_type == "EA" else self.step_hybrid
        means = np.empty(generations, dtype=np.float64)
        bests = np.empty(generations, dtype=np.float64)
        pbar = tqdm(range(generations), desc="Generations", unit="gen")
        for g in pbar:
            best_fit,_ = step_fn()
            if self._last_fits is None:
                raise RuntimeError("Internal error: fits not recorded this generation.")
            fits = self._last_fits
            means[g] = float(np.mean(fits))
            bests[g] = best_fit
            pbar.set_postfix(best_fit=f"{self._best_fit: .4f}")
        pbar.close()
        self.fitness_history = {"mean": means, "best": bests}
        self.shutdown_ray()
        return self._best_x.copy(), self._best_fit # type: ignore 
    def shutdown_ray(self) -> None:
        """Shut down Ray runtime if it was started by this optimizer."""
        if self.use_ray and ray.is_initialized():
            ray.shutdown()

    # string representation -----------------------------------------
    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"PureNOMAD(mu={self.mu}, D={self.D}, gen={self.generation}, "
            f"best={self._best_fit:.3e})"
        )
if _RAY_AVAILABLE:
    @ray.remote
    class ObjectiveActor:
        def __init__(self, fn: Callable[[npt.NDArray], float]):
            self.fn = fn

        def eval(self, x: npt.NDArray) -> float:
            return self.fn(x)

        def nomad(self, x0, full_x, idx, max_bb_eval, bounds=None, lb=None, ub=None):
            return self._run_nomad(self.fn, x0, full_x, idx, max_bb_eval, bounds, lb, ub)

        @staticmethod
        def _run_nomad(fn, x0, full_x, idx, max_bb_eval, bounds, lb, ub):
            return _nomad_local_search(fn, x0, full_x, idx, max_bb_eval, bounds, lb, ub)
