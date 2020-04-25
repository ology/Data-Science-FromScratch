#!/usr/bin/env perl
use strict;
use warnings;

use Test::More;

use_ok 'Data::Science::FromScratch';

my $ds = new_ok 'Data::Science::FromScratch';

# Vector methods

is_deeply $ds->vector_sum([1,2,3], [4,5,6]), [5,7,9], 'vector_sum';
is_deeply $ds->vector_sum([1,2], [3,4], [5,6], [7,8]), [16,20], 'vector_sum';
is_deeply $ds->vector_sum([1,2], [3,4], [5,6]), [9,12], 'vector_sum';

is_deeply $ds->vector_subtract([5,7,9], [4,5,6]), [1,2,3], 'vector_subtract';

is_deeply $ds->scalar_multiply(2, [1,2,3]), [2,4,6], 'scalar_multiply';

is_deeply $ds->vector_mean([1,2], [3,4], [5,6]), [3,4], 'vector_mean';

is $ds->vector_dot([1,2,3], [4,5,6]), 32, 'vector_dot';

is $ds->sum_of_squares([1,2,3]), 14, 'sum_of_squares';

is $ds->magnitude([3,4]), 5, 'magnitude';

is $ds->distance([0,1], [1,1]), 1, 'distance';

# Matrix methods

is_deeply $ds->shape([[1,2,3], [4,5,6]]), [2,3], 'shape';

is_deeply $ds->get_row([[1,2,3], [4,5,6]], 0), [1,2,3], 'get_row';
is_deeply $ds->get_col([[1,2,3], [4,5,6]], 0), [1,4], 'get_col';

is_deeply $ds->make_matrix(2, 3, sub { 0 }),
    [[0,0,0], [0,0,0]], 'make_matrix';
is_deeply $ds->make_matrix(3, 3, sub { my ($i, $j) = @_; return $i == $j ? 1 : 0 }),
    [[1,0,0], [0,1,0], [0,0,1]], 'make_matrix';

done_testing();
