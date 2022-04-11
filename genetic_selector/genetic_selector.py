# genetic_selector - Genetic algorithm for feature selection
# Juan Carlos Ruiz García - September 2020


import multiprocessing as mp
import numbers
import time

import numpy as np
import pandas as pd
from sklearn.metrics import get_scorer
from sklearn.base import is_classifier
from sklearn.model_selection import cross_validate
from sklearn.model_selection._split import check_cv

"""
Genetic algorithm used to select the best features from a dataset
"""


class GeneticSelector:
    """Feature selection with genetic algorithm.
    Parameters
    --------------------
    estimator : object
        A supervised learning estimator with a `fit` method from
        Scikit-learn.
    scoring : str, callable, or None (default: None)
        If None (default), uses 'accuracy' for sklearn classifiers
        and 'r2' for sklearn regressors.
        If str, uses a sklearn scoring metric string identifier, for example
        {accuracy, f1, precision, recall, roc_auc} for classifiers,
        {'mean_absolute_error', 'mean_squared_error'/'neg_mean_squared_error',
        'median_absolute_error', 'r2'} for regressors.
    cv : int, cross-validation generator or iterable, default=None
        Determines the cross-validation splitting strategy.
        Possibilities are:

        - None, to use the default 5-fold cross validation,
        - int, to specify the number of folds in a `(Stratified)KFold`,
        - :term:`CV splitter`,
        - An iterable yielding (train, test) splits as arrays of indices.
    n_gen : int, default: 50
        Determines the maximum number of generations to be carry out.
    population_size : int, default: 100
        Determines the size of the population (number of chromosomes).
    crossover_rate : float, default: 0.7
        Defines the crossing probability. It must be a value between 0.0 and
        1.0.
    mutation_rate : double, default: 0.1
        Defines the mutation probability. It must be a value between 0.0 and
        1.0.
    tournament_k : int, default: 2
        Defines the size of the tournament carried out in the selection
        process. Number of chromosomes facing each other in each tournament.
    calc_train_score : bool, default=False
        Whether or not to calculate the scores obtained during the training
        process. The calculation of training scores is used to obtain
        information on how different parameter settings affect the
        overfitting/underfitting trade-off. However, calculating the scores in
        the training set can be computationally expensive and is not strictly
        necessary to select the parameters that produce the best generalisation
        performance.
    initial_best_chromosome: np.ndarray, default=None
        A 1-dimensional binary matrix of size equal to the number of features
        (M). Defines the best chromosome (subset of features) in the initial
        population.
    n_jobs : int, default 1
        Number of cores to run in parallel.
        By default a single-core is used. `n_jobs`=-1 means the maximum number
        of cores on the machine. If the inserted `n_jobs` is greater than the
        maximum number of cores on the machine, then the value is set to the
        maximum number of cores on the machine.
    random_state : int or RandomState instance, default=None
        Controls the randomness of the life cycle in each population. Enter an
        integer for reproducible output.
    verbose : int, default=0
        Control the output verbosity level. It must be an integer value between
        0 and 2.
    """

    def __init__(self, estimator: object, scoring: str = None, cv: int = 5,
                 n_gen: int = 50, population_size: int = 100, crossover_rate:
                 float = 0.7, mutation_rate: float = 0.1,
                 tournament_k: int = 2, calc_train_score: bool = False,
                 initial_best_chromosome: np.ndarray = None, n_jobs: int = 1,
                 random_state: int = None, verbose: int = 0):

        # Sklearn estimator (classifier or regressor)
        self.estimator = estimator
        # Save train score
        self.calc_train_score = calc_train_score
        # Initial chromosome
        self.initial_best_chromosome = initial_best_chromosome
        # Scoring metric
        if scoring is None:
            if self.estimator._estimator_type == 'classifier':
                scoring = 'accuracy'
            elif self.est_._estimator_type == 'regressor':
                scoring = 'r2'
            else:
                raise AttributeError('Estimator must '
                                     'be a Classifier or Regressor.')
        if isinstance(scoring, str):
            self.scorer = get_scorer(scoring)
        else:
            self.scorer = scoring
        # Number of folds in cross validation process
        self.cv = cv
        # Number of generations
        if n_gen > 0:
            self.n_gen = n_gen
        else:
            raise ValueError(
                'The number of generations must be greater than 1.')
        # Size of population (number of chromosomes)
        self.population_size = population_size
        # Crossover and mutations likelihood
        if crossover_rate <= 0.0 or mutation_rate <= 0.0 or \
                crossover_rate > 1.0 or mutation_rate > 1.0:
            raise ValueError(
                'Mutation and crossover rate must be a value in the range'
                ' (0.0, 1.0].')
        else:
            self.mutation_rate = mutation_rate
            self.crossover_rate = crossover_rate
        # Tournament size in selection process
        self.tournament_k = tournament_k
        # Number of threads
        if n_jobs < 0 and n_jobs == -1:
            self.n_jobs = mp.cpu_count()
        elif n_jobs > 0 and n_jobs <= mp.cpu_count():
            self.n_jobs = n_jobs
        else:
            raise ValueError(
                f'n_jobs == {n_jobs} is invalid! You have a maximum of'
                f' {mp.cpu_count()} cores.')
        # Best chromosome (np.ndarray, float, int) (chromosome, score, i_gen)
        self.best_chromosome = (None, None, None)
        # Population convergence variables
        self.convergence = False
        self.n_times_convergence = 0
        self.threshold = 1e-6
        # Random state
        if isinstance(random_state, numbers.Integral):
            np.random.seed(random_state)
        # Verbose
        if verbose < 0:
            self.verbose = 0
        else:
            self.verbose = verbose

    def fit(self, X: pd.DataFrame, y: pd.Series):
        # Initialize output variables
        self.train_scores = []
        self.val_scores = []
        self.chromosomes_history = []

        # Check cross-validation is valid
        self.cv = check_cv(self.cv, y, classifier=is_classifier(
            self.estimator))

        # Time when training begins
        init_time = time.time()

        # Save data and targets in the instance
        self.X = X
        self.y = y

        # Initialize population
        print(
            f'# Creating initial population with {self.population_size}'
            ' chromosomes...')
        self.__initialize(X.shape[1])

        # Insert initial best chromosome if is defined
        if self.initial_best_chromosome is None:

            # Evaluate initial population and update best_chromosome
            print(f'# Evaluating initial population...')
            population_scores = self.__evaluate(self.population)
            best_chromosome_index = np.argmax(population_scores[:, 0])

        elif type(self.initial_best_chromosome) == np.ndarray and \
            len(self.initial_best_chromosome) == X.shape[1] and \
            len(np.where((self.initial_best_chromosome <= 1) &
                         (self.initial_best_chromosome >= 0))[0]):

            # Introduce the best in the population
            index_insertion = np.random.randint(self.population_size)
            self.population[index_insertion] = self.initial_best_chromosome

            # Evaluate initial population and update best_chromosome
            print(f'# Evaluating initial population...')
            population_scores = self.__evaluate(self.population)
            best_chromosome_index = index_insertion

        else:
            raise ValueError('Initial best chromosome must be a 1 '
                             'dimensional binary array with a length of '
                             'X.shape[1]')

        # Update best chromosome found
        self.__update_best_chromosome(
            self.population, population_scores[:, 0],
            best_chromosome_index, 0)

        # Save output results
        if self.calc_train_score:
            self.__save_output_results(
                val_score=population_scores[best_chromosome_index, 0],
                best_current_chromosome=self.population[best_chromosome_index],
                train_score=population_scores[best_chromosome_index, 1])
        else:
            self.__save_output_results(
                val_score=population_scores[best_chromosome_index, 0],
                best_current_chromosome=self.population[best_chromosome_index])

        # Loop until evaluation converge
        i = 0
        while i < self.n_gen and not self.convergence:
            # Time when generation begins
            generation_time = time.time()

            print(f'\n# Creating generation {i+1}...')
            # Selection
            new_population = self.__selection(
                self.population, population_scores[:, 0],
                best_chromosome_index)
            print(f'# Selection {i+1} done.')

            # Crossover
            new_population = self.__crossover(new_population)
            print(f'# Crossover {i+1} done.')

            # Mutation
            new_population = self.__mutation(new_population)
            print(f'# Mutation {i+1} done.')

            # Replace previous population with new_population
            self.population = new_population.copy()

            # Evaluate new population and update best_chromosome
            print(f'# Evaluating population of new generation {i}...')
            population_scores = self.__evaluate(self.population)
            best_chromosome_index = np.argmax(population_scores[:, 0])
            self.__update_best_chromosome(
                self.population, population_scores[:, 0],
                best_chromosome_index, i+1)

            # Save output results
            if self.calc_train_score:
                self.__save_output_results(
                    val_score=population_scores[best_chromosome_index, 0],
                    best_current_chromosome=self.population
                    [best_chromosome_index],
                    train_score=population_scores[best_chromosome_index, 1])
            else:
                self.__save_output_results(
                    val_score=population_scores[best_chromosome_index, 0],
                    best_current_chromosome=self.population
                    [best_chromosome_index])

            # Next generation
            i = i + 1

            # Time when generation ends
            elapsed_generation_time = time.time() - generation_time
            print(
                '# Elapsed generation time: %.2f seconds' %
                elapsed_generation_time)

        # Time when training ends
        elapsed_time = time.time() - init_time
        print('# Elapsed time: %.2f seconds' % elapsed_time)

    def __save_output_results(self, val_score: float,
                              best_current_chromosome: np.ndarray,
                              train_score: float = None):
        """

        Private function to save the output results in their respective
        variables.

        Args:
            val_score (float): Best validation score achieved in the present
            generation for the best chromosome.
            train_score (float): Trainig score achieved in the present
            generation for the best chromosome.
            best_current_chromosome (np.ndarray): That chromosome whose
                val_score was the best one found in the present generation.
        """
        self.val_scores.append(val_score)
        if train_score is not None:
            self.train_scores.append(train_score)
        self.chromosomes_history.append(best_current_chromosome)

    def support(self) -> np.ndarray:
        """
            Return an array with 4 values:
                - best_chromosome : tuple
                    Tuple with the values (np.ndarray, float, int) =>
                    (chromosome, score, i_gen)
                - val_scores : np.ndarray
                    An array with validation scores during each generation.
                - train_scores : np.ndarray
                    An array with training scores during each generation. Could
                    be `None` if self.calc_train_score = False.
                - chromosomes_history : np.ndarray
                    An array with multiple mask of selected features, each one
                    for the best chromosome found in each generation.
        """

        return np.array([
            self.best_chromosome,
            self.val_scores,
            self.train_scores if self.calc_train_score else None,
            self.chromosomes_history
        ])

    def __initialize(self, n_genes: int):
        # Create population_size chromosomes
        self.population = np.random.randint(
            2, size=(self.population_size, n_genes))

    def estimate(self, chromosome: np.ndarray) -> float:
        # Select those features with ones in chromosome
        data = self.X.loc[:, chromosome.astype(bool)]

        # Cross-validation execution
        scores = cross_validate(
            self.estimator, data, self.y, cv=self.cv, scoring=self.scorer,
            verbose=self.verbose, return_train_score=self.calc_train_score)

        if self.calc_train_score:
            return (scores['test_score'].mean(),
                    scores['train_score'].mean())
        else:
            return (scores['test_score'].mean(), None)

    def __evaluate(self, population: np.ndarray) -> np.ndarray:
        # Pool for parallelization
        pool = mp.Pool(self.n_jobs)
        return np.array(pool.map(self.estimate, population))

    def __selection(
            self, population: np.ndarray, population_scores: np.ndarray,
            best_chromosome_index: int) -> np.ndarray:
        # Create new population
        new_population = [population[best_chromosome_index]]

        # Tournament_k chromosome tournament until fill the numpy array
        while len(new_population) != self.population_size:
            # Generate tournament_k positions randomly
            k_chromosomes = np.random.randint(
                len(population), size=self.tournament_k)
            # Get the best one of these tournament_k chromosomes
            best_of_tournament_index = np.argmax(
                population_scores[k_chromosomes])
            # Append it to the new population
            new_population.append(
                population[k_chromosomes[best_of_tournament_index]])

        return np.array(new_population)

    def __crossover(self, population: np.ndarray) -> np.ndarray:
        # Define the number of crosses
        n_crosses = int(self.crossover_rate * int(self.population_size / 2))

        # Make a copy from current population
        crossover_population = population.copy()

        # Make n_crosses crosses
        for i in range(0, n_crosses*2, 2):
            cut_index = np.random.randint(1, self.X.shape[1])
            tmp = crossover_population[i, cut_index:].copy()
            crossover_population[i, cut_index:], crossover_population[i+1,
                                                                      cut_index:] = crossover_population[i+1, cut_index:], tmp
            # Avoid null chromosomes
            if not all(crossover_population[i]):
                crossover_population[i] = population[i]
            if not all(crossover_population[i+1]):
                crossover_population[i+1] = population[i+1]

        return crossover_population

    def __mutation(self, population: np.ndarray) -> np.ndarray:
        # Define number of mutations to do
        n_mutations = int(
            self.mutation_rate * self.population_size * self.X.shape[1])

        # Mutating n_mutations genes
        for _ in range(n_mutations):
            chromosome_index = np.random.randint(0, self.population_size)
            gene_index = np.random.randint(0, self.X.shape[1])
            population[chromosome_index, gene_index] = 0 if \
                population[chromosome_index, gene_index] == 1 else 1

        return population

    def __update_best_chromosome(
            self, population: np.ndarray, population_scores: np.ndarray,
            best_chromosome_index: int, i_gen: int):
        # Initialize best_chromosome
        if self.best_chromosome[0] is None and self.best_chromosome[1] is None:
            self.best_chromosome = (
                population[best_chromosome_index],
                population_scores[best_chromosome_index],
                i_gen)
            self.threshold_times_convergence = 5
        # Update if new is better
        elif population_scores[best_chromosome_index] > \
                self.best_chromosome[1]:
            if i_gen >= 17:
                self.threshold_times_convergence = int(np.ceil(0.3 * i_gen))
            self.best_chromosome = (
                population[best_chromosome_index],
                population_scores[best_chromosome_index],
                i_gen)
            self.n_times_convergence = 0
            print(
                '# (BETTER) A better chromosome than the current one has '
                f'been found ({self.best_chromosome[1]}).')
        # If is smaller than self.threshold count it until convergence
        elif abs(population_scores[best_chromosome_index] -
                 self.best_chromosome[1]) <= self.threshold:
            self.n_times_convergence = self.n_times_convergence + 1
            print(
                f'# Same scoring value found {self.n_times_convergence}'
                f'/{self.threshold_times_convergence} times.')
            if self.n_times_convergence == self.threshold_times_convergence:
                self.convergence = True
        else:
            self.n_times_convergence = 0
            print('# (WORST) No better chromosome than the current one has '
                  f'been found ({population_scores[best_chromosome_index]}).')
        print(f'# Current best chromosome: {self.best_chromosome}')
