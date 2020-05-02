#!/usr/bin/env perl
use strict;
use warnings;

use Test::More;

use Data::Dataset::Classic::Iris;

use_ok 'Data::Science::FromScratch';

my $ds = new_ok 'Data::Science::FromScratch';

is $ds->step_function(-1), 0, 'step_function';
is $ds->step_function(0), 1, 'step_function';
is $ds->step_function(1), 1, 'step_function';

done_testing();
