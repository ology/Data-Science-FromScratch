#!/usr/bin/env perl
use strict;
use warnings;

use Test::More;

use_ok 'Data::Science::FromScratch';

my $ds = new_ok 'Data::Science::FromScratch';

is_deeply $ds->tensor_shape([1,2,3]), [3], 'tensor_shape';
is_deeply $ds->tensor_shape([[1,2],[3,4],[5,6]]), [3,2], 'tensor_shape';

ok $ds->is_1d([1,2,3]), 'is_1d';
ok ! $ds->is_1d([[1,2],[3,4]]), 'is_1d';

is $ds->tensor_sum([1,2,3]), 6, 'tensor_sum';
is $ds->tensor_sum([[1,2],[3,4]]), 10, 'tensor_sum';

is_deeply $ds->tensor_apply(sub { shift() + 1 }, [1,2,3]), [2,3,4], 'tensor_apply';
is_deeply $ds->tensor_apply(sub { 2 * shift() }, [[1,2],[3,4]]), [[2,4], [6,8]], 'tensor_apply';

is_deeply $ds->zeros_like([1,2,3]), [0,0,0], 'zeros_like';
is_deeply $ds->zeros_like([[1,2],[3,4]]), [[0,0],[0,0]], 'zeros_like';

is_deeply $ds->tensor_combine(sub { shift() + shift() }, [1,2,3], [4,5,6]), [5,7,9], 'tensor_combine';
is_deeply $ds->tensor_combine(sub { shift() * shift() }, [1,2,3], [4,5,6]), [4,10,18], 'tensor_combine';

use_ok 'Data::Science::FromScratch::Sigmoid';

my $sigmoid = new_ok 'Data::Science::FromScratch::Sigmoid';

is_deeply $sigmoid->forward([0,0,0]), [0.5,0.5,0.5], 'forward';
is_deeply $sigmoid->backward([0,0,0]), [0,0,0], 'backward';
is_deeply $sigmoid->backward([1,2,3]), [0.25,0.5,0.75], 'backward';

is_deeply $ds->tensor_shape($ds->random_uniform([2,3,4])), [2,3,4], 'random_uniform';
is_deeply $ds->tensor_shape($ds->random_normal([5,6], 10)), [5,6], 'random_normal';

use_ok 'Data::Science::FromScratch::Linear';

my $linear = new_ok 'Data::Science::FromScratch::Linear' => [
    output_dim => 3,
    input_dim  => 2,
];

done_testing();
