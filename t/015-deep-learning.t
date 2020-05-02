#!/usr/bin/env perl
use strict;
use warnings;

use Test::More;

use Data::Dataset::Classic::Iris;

use_ok 'Data::Science::FromScratch';

my $ds = new_ok 'Data::Science::FromScratch';

is_deeply $ds->dl_shape([1,2,3]), [3], 'dl_shape';
is_deeply $ds->dl_shape([[1,2],[3,4],[5,6]]), [3,2], 'dl_shape';

done_testing();
