#!/usr/bin/env perl
use strict;
use warnings;

use Test::More;

use_ok 'Data::MachineLearning::Elements';

my $ml = new_ok 'Data::MachineLearning::Elements';

# Vector methods

is_deeply $ml->vector_sum([1,2,3], [4,5,6]), [5,7,9], 'vector_sum';
is_deeply $ml->vector_sum([1,2], [3,4], [5,6], [7,8]), [16,20], 'vector_sum';
is_deeply $ml->vector_sum([1,2], [3,4], [5,6]), [9,12], 'vector_sum';

is_deeply $ml->vector_subtract([5,7,9], [4,5,6]), [1,2,3], 'vector_subtract';

is_deeply $ml->scalar_multiply(2, [1,2,3]), [2,4,6], 'scalar_multiply';

is_deeply $ml->vector_mean([1,2], [3,4], [5,6]), [3,4], 'vector_mean';
is_deeply $ml->vector_mean([2,4], [6,8]), [4,6], 'vector_mean';

is $ml->vector_dot([1,2,3], [4,5,6]), 32, 'vector_dot';

is $ml->sum_of_squares([1,2,3]), 14, 'sum_of_squares';

is $ml->magnitude([3,4]), 5, 'magnitude';

is $ml->distance([0,1], [1,1]), 1, 'distance';

is $ml->squared_distance([0,1], [1,1]), 1, 'squared_distance';
is $ml->squared_distance([0,0], [2,2]), 8, 'squared_distance';

# Matrix methods

is_deeply $ml->shape([[1,2,3], [4,5,6]]), [2,3], 'shape';

is_deeply $ml->get_row([[1,2,3], [4,5,6]], 0), [1,2,3], 'get_row';
is_deeply $ml->get_col([[1,2,3], [4,5,6]], 0), [1,4], 'get_col';

is_deeply $ml->make_matrix(2, 3, sub { 0 }),
    [[0,0,0], [0,0,0]], 'make_matrix';
is_deeply $ml->make_matrix(3, 3, sub { my ($i, $j) = @_; return $i == $j ? 1 : 0 }),
    [[1,0,0], [0,1,0], [0,0,1]], 'make_matrix';

done_testing();
