#!/usr/bin/env perl
use strict;
use warnings;

use Test::More;

use Data::Dataset::Classic::Iris;

use_ok 'Data::Science::FromScratch';

my $ds = new_ok 'Data::Science::FromScratch';

is_deeply $ds->dl_shape([1,2,3]), [3], 'dl_shape';
is_deeply $ds->dl_shape([[1,2],[3,4],[5,6]]), [3,2], 'dl_shape';

ok $ds->is_1d([1,2,3]), 'is_1d';
ok ! $ds->is_1d([[1,2],[3,4]]), 'is_1d';

is $ds->tensor_sum([1,2,3]), 6, 'tensor_sum';
is $ds->tensor_sum([[1,2],[3,4]]), 10, 'tensor_sum';

done_testing();
