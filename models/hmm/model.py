import itertools
import numpy as np
from nltk import ngrams
from datetime import datetime


class HiddenMarkovModel:

    def __init__(self, order):
        self.order = order
        self.left_pad = '<s>'
        self.right_pad = '<S>'
        # --------------------------------      -                               -
        self.observations = None                # unique observations
        self.states = None                      # unique states
        self.pi = None                          # initial probability matrix
        self.A = None                           # transition probability matrix
        self.B = None                           # emission probability matrix
        # -------------------------             -                               -
        self.population = None                  # dict<observation, possible states<count>>
        self.ngrams = None                      # dict<ngram, count>
        self.smoothing = None                   # None or 'one-count'
        # -------------------------             -                               -
        self.prior_A = None
        self.prior_B = None
        self.singleton_transition = None
        self.singleton_emission = None

    # ============== main functions =============

    def fit(self, X, y, smoothing=None):
        """
        Builds a hmm based on the given train sequences.

        Args:
            X (list<list<str>>): Observation sequences.
            y (list<list<str>>): State sequences.
            smoothing (str): Smoothing method. Default None. Support 'one-count'.
        """

        if smoothing and smoothing != 'one-count':
            raise ValueError

        self.smoothing = smoothing

        # build observation and state dicts
        self.observations = self._value_counts(X)
        self.states = self._value_counts(y)

        # build observation population and ngram dicts
        self.population = self._population(X, y)
        self.ngrams = self._ngrams(y, self.order, self.left_pad, self.right_pad)
        self.prior_A = self._prior_transition(self.states)
        self.prior_B = self._prior_emission(self.population, sum(self.states.values()))

        # build probability distribution parameters
        if smoothing == 'one-count':
            self.singleton_transition, self.singleton_emission = self._singletons(self.ngrams, self.population)
            self.A = self._oc_transition_probabilities(self.ngrams, self.order)
            self.B = self._oc_emission_probabilities(self.population)
        else:
            self.A = self._transition_probabilities(self.ngrams, self.order)
            self.B = self._emission_probabilities(self.population, self.states)
            self.pi = self._start_probabilities(self.A, self.order)

    def predict(self, X, verbose=False):
        """
        Predicts state sequences by given observation. Uses Viterbi algorithm.

        Args:
            X (list<list<str>>): Observation sequence.
            verbose (bool): Default False. Enable verbose output.
        Return:
            y_pred (list<list<str>>): State sequence.
        """
        y_pred = []
        for i, sequence in enumerate(X):
            if verbose:
                print(f'\n[{datetime.now().strftime("%H:%M:%S")}] HMM: predicting {i} | {len(X)}')
            if not sequence:
                y_pred.append([])
            else:
                y_pred.append(self._viterbi_algorithm(sequence))
        return y_pred

    def forward_probability(self, X, y):
        """
        Finds path probability.

        Args:
            X (list<list<str>>): Observation sequences.
            y (list<list<str>>): State sequences.
        """
        probs = []
        for X_i, y_i in zip(X, y):
            prob = 0
            X, y = [*(self.left_pad,) * (self.order - 1), *X_i], [*(self.left_pad,) * (self.order - 1), *y_i]
            for i, (obs, state) in enumerate(zip(X[self.order-1:], y[self.order-1:]), start=0):
                prob += self._get_transition(tuple(y[i:i + self.order])) + self._get_emission(obs, state)
            probs.append(prob)
        return probs

    # ============= model parameters ============

    def set_parameters(self, A, B, pi, observations, states):
        self.A = A
        self.B = B
        self.pi = pi
        self.observations = observations
        self.states = states

    def get_parameters(self):
        return {
            'a': self.A,
            'b': self.B,
            'pi': self.pi,
            'observations': self.observations,
            'states': self.states
        }

    # ================== utils ==================

    @staticmethod
    def _value_counts(sequences):
        value_counts = dict()
        for sequence in sequences:
            for element in sequence:
                value_counts[element] = value_counts.get(element, 0) + 1
        return value_counts

    @staticmethod
    def _population(obs_sequences, state_sequences):
        population = dict()
        for obs_sequence, state_sequence in zip(obs_sequences, state_sequences):
            for observation, state in zip(obs_sequence, state_sequence):
                if observation not in population:
                    population[observation] = dict()
                population[observation][state] = population[observation].get(state, 0) + 1
        return population

    @staticmethod
    def _ngrams(sequences, n, left_pad='<s>', right_pad='<S>'):
        counts = dict()
        for sequence in sequences:
            sequence = [*(left_pad,)*(n-1), *sequence, right_pad]
            for i in range(n, 0, -1):
                for ngram in ngrams(sequence, i):
                    counts[ngram] = counts.get(ngram, 0) + 1
        return counts

    # =========== calculate parameters ==========

    @staticmethod
    def _prior_transition(states):
        prior_transitions = dict()
        for state in states.keys():
            prior_transitions[state] = (1 + states[state]) / (len(states) + sum(states.values()))
        return prior_transitions

    @staticmethod
    def _prior_emission(population, size):
        prior_emission = dict()
        for observation, states in population.items():
            prior_emission[observation] = sum(states.values()) / size
        return prior_emission

    @staticmethod
    def _start_probabilities(probabilities, n, left_pad='<s>'):
        """
        Calculate initial probabilities by the given train state sequences.

        Args:
            probabilities (dict<tuple><float>):
            left_pad (str):
        Returns:
            Initial probability distribution
        """
        start_probabilities = dict()
        for ngram, probability in probabilities.items():
            if ngram[:-1] == (left_pad,)*(n-1):
                start_probabilities[ngram] = probability
        return start_probabilities

    @staticmethod
    def _transition_probabilities(grams, n):
        transition_probabilities = dict()
        for gram in grams.keys():
            if len(gram) == n:
                transition_probabilities[gram] = grams[gram] / grams[gram[:-1]]
        return transition_probabilities

    @staticmethod
    def _emission_probabilities(population, states):
        emission_probabilities = dict()
        for observation, obs_states in population.items():
            if observation not in emission_probabilities:
                emission_probabilities[observation] = dict()
            for state in obs_states:
                emission_probabilities[observation][state] = population[observation][state] / states[state]
        return emission_probabilities

    # ========== one-count parameters ===========

    @staticmethod
    def _singletons(grams, population):
        s_transition = dict()
        for gram, count in grams.items():
            length = len(gram)
            if count == 1 and length > 1:
                s_transition[gram[:length-1]] = s_transition.get(gram[:length-1], 0) + 1
        s_emission = dict()
        for observation, states in population.items():
            for state, count in states.items():
                if count == 1:
                    s_emission[state] = s_emission.get(state, 0) + 1
        return s_transition, s_emission

    def _get_oc_transition(self, s_tuple):
        length = len(s_tuple)
        if length == 1:
            return self.prior_A.get(s_tuple[0], 1e-6)
        else:
            alpha = 1 + self.singleton_transition.get(s_tuple[:length-1], 0)
            p = ((self.ngrams.get(s_tuple, 0) + alpha * self._get_oc_transition(s_tuple[-(length-1):])) /
                 (self.ngrams.get(s_tuple[:length-1], 0) + alpha))
        return p

    def _get_oc_emission(self, observation, state):
        beta = 1 + self.singleton_emission.get(state, 0)
        p = ((self.population.get(observation, {}).get(state, 0) + beta *
              self.prior_B.get(observation, 1 / (sum(self.states.values()) + len(self.states)))) /
             (self.ngrams[(state,)] + beta))
        return p

    def _oc_transition_probabilities(self, grams, n):
        transition_probabilities = dict()
        for gram in grams.keys():
            if len(gram) == n:
                transition_probabilities[gram] = self._get_oc_transition(gram)
        return transition_probabilities

    def _oc_emission_probabilities(self, population):
        emission_probabilities = dict()
        for observation, states in population.items():
            if observation not in emission_probabilities:
                emission_probabilities[observation] = dict()
            for state in states:
                emission_probabilities[observation][state] = self._get_oc_emission(observation, state)
        return emission_probabilities

    # ========== get single parameters ==========

    def _get_states(self, observation):
        if observation == self.left_pad:
            return [self.left_pad]
        elif observation == self.right_pad:
            return [self.right_pad]
        elif observation in self.population:
            return self.population[observation].keys()
        return list(self.states)

    def _get_initial(self, state):
        s_tuple = (*(self.left_pad,)*(self.order-1), state)
        if s_tuple in self.pi:
            return np.log(self.pi[s_tuple])
        elif self.smoothing == 'one-count':
            return np.log(self._get_oc_transition(s_tuple))
        else:
            return np.log(1e-18)

    def _get_transition(self, s_tuple):
        if s_tuple in self.A:
            return np.log(self.A[s_tuple])
        elif self.smoothing == 'one-count':
            return np.log(self._get_oc_transition(s_tuple))
        else:
            return np.log(1e-18)

    def _get_emission(self, observation, state):
        if observation in self.B and state in self.B[observation]:
            return np.log(self.B[observation][state])
        elif self.smoothing == 'one-count':
            return np.log(self._get_oc_emission(observation, state))
        else:
            return np.log(1e-18)

    # ============ viterbi algorithm ============

    def _viterbi_algorithm(self, sequence):
        pi, bp = self._viterbi_forward(sequence)
        return self._viterbi_backward([self.left_pad, *sequence], pi, bp)

    def _viterbi_tuples(self, observations):
        states = []
        for i, observation in enumerate(reversed(observations)):
            states.append(self._get_states(observation))
        products = itertools.product(*states)
        rev_products = [product[::-1] for product in products]
        return rev_products

    def _viterbi_forward(self, sequence):
        bp = dict()
        pi = dict({(-1, *(self.left_pad,) * (self.order - 1)): 0})
        sentence = [*(self.left_pad,) * (self.order - 1), *sequence, self.right_pad]
        for step, word in enumerate(sentence[self.order-1:-1], start=0):
            tuples = self._viterbi_tuples(sentence[step:step+self.order])
            tuples_step = len(self._get_states(sentence[step]))
            for i in range(0, len(tuples), tuples_step):
                scores = []
                for t in tuples[i:i+tuples_step]:
                    transition = self._get_transition(t)
                    emission = self._get_emission(word, t[-1])
                    scores.append(pi[(step-1, *t[:self.order-1])] + transition + emission)
                m_tuple = tuples[i:i+tuples_step][np.argmax(scores)]
                pi[(step, *m_tuple[-(self.order-1):])] = np.max(scores)
                bp[(step, *m_tuple[-(self.order-1):])] = m_tuple[:self.order-1]
        return pi, bp

    def _viterbi_backward(self, sequence, pi, bp):
        back_pointers = []
        for step in reversed(range(len(sequence))):
            if step == len(sequence) - 1:
                scores = []
                tuples = self._viterbi_tuples([*sequence[step-self.order+2:], self.right_pad])
                for t in tuples:
                    transition = self._get_transition(t)
                    scores.append(pi[(step-1, *t[:-1])] + transition)
                m_tuple = tuples[np.argmax(scores)]
                back_pointers.append(m_tuple[:-1])
            else:
                back_pointers.insert(0, bp[(step, *back_pointers[0])])
        tag_sequence = [b[-1] for b in back_pointers[1:]]
        return tag_sequence
