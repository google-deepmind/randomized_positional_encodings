# Copyright 2023 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Constants for our length generalization experiments."""

import functools

from randomized_positional_encodings.experiments import curriculum as curriculum_lib
from randomized_positional_encodings.models import transformer
from randomized_positional_encodings.tasks.cs import binary_addition
from randomized_positional_encodings.tasks.cs import binary_multiplication
from randomized_positional_encodings.tasks.cs import bucket_sort
from randomized_positional_encodings.tasks.cs import compute_sqrt
from randomized_positional_encodings.tasks.cs import duplicate_string
from randomized_positional_encodings.tasks.cs import missing_duplicate_string
from randomized_positional_encodings.tasks.cs import odds_first
from randomized_positional_encodings.tasks.dcf import modular_arithmetic_brackets
from randomized_positional_encodings.tasks.dcf import reverse_string
from randomized_positional_encodings.tasks.dcf import solve_equation
from randomized_positional_encodings.tasks.dcf import stack_manipulation
from randomized_positional_encodings.tasks.regular import cycle_navigation
from randomized_positional_encodings.tasks.regular import even_pairs
from randomized_positional_encodings.tasks.regular import modular_arithmetic
from randomized_positional_encodings.tasks.regular import parity_check


MODEL_BUILDERS = {
    'transformer_encoder': functools.partial(
        transformer.make_transformer,
        transformer_module=transformer.TransformerEncoder,  # pytype: disable=module-attr
    ),
}

CURRICULUM_BUILDERS = {
    'fixed': curriculum_lib.FixedCurriculum,
    'regular_increase': curriculum_lib.RegularIncreaseCurriculum,
    'reverse_exponential': curriculum_lib.ReverseExponentialCurriculum,
    'uniform': curriculum_lib.UniformCurriculum,
}

TASK_BUILDERS = {
    'even_pairs': even_pairs.EvenPairs,
    'modular_arithmetic': functools.partial(
        modular_arithmetic.ModularArithmetic, modulus=5
    ),
    'parity_check': parity_check.ParityCheck,
    'cycle_navigation': cycle_navigation.CycleNavigation,
    'stack_manipulation': stack_manipulation.StackManipulation,
    'reverse_string': functools.partial(
        reverse_string.ReverseString, vocab_size=2
    ),
    'modular_arithmetic_brackets': functools.partial(
        modular_arithmetic_brackets.ModularArithmeticBrackets,
        modulus=5,
        mult=True,
    ),
    'solve_equation': functools.partial(
        solve_equation.SolveEquation, modulus=5
    ),
    'duplicate_string': functools.partial(
        duplicate_string.DuplicateString, vocab_size=2
    ),
    'missing_duplicate_string': missing_duplicate_string.MissingDuplicateString,
    'odds_first': functools.partial(odds_first.OddsFirst, vocab_size=2),
    'binary_addition': binary_addition.BinaryAddition,
    'binary_multiplication': binary_multiplication.BinaryMultiplication,
    'compute_sqrt': compute_sqrt.ComputeSqrt,
    'bucket_sort': functools.partial(bucket_sort.BucketSort, vocab_size=5),
}

TASK_LEVELS = {
    'even_pairs': 'regular',
    'modular_arithmetic': 'regular',
    'parity_check': 'regular',
    'cycle_navigation': 'regular',
    'stack_manipulation': 'dcf',
    'reverse_string': 'dcf',
    'modular_arithmetic_brackets': 'dcf',
    'solve_equation': 'dcf',
    'duplicate_string': 'cs',
    'missing_duplicate_string': 'cs',
    'odds_first': 'cs',
    'binary_addition': 'cs',
    'binary_multiplication': 'cs',
    'compute_sqrt': 'cs',
    'bucket_sort': 'cs',
}
