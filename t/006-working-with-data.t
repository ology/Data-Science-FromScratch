#!/usr/bin/env perl
use strict;
use warnings;

use Test::More;

use_ok 'Data::Science::FromScratch';

my $ds = new_ok 'Data::Science::FromScratch';

my @data = ([-3, -1, 1], [-1, 0, 1], [1, 1, 1]);

my ($x, $y) = $ds->scale(@data);
is_deeply $x, [-1, 0, 1], 'scale means';
is_deeply $y, [2, 1, 0], 'scale stddev';

($x, $y) = $ds->scale($ds->rescale(@data));
is_deeply $x, [0, 0, 1], 'scale means';
is_deeply $y, [1, 1, 0], 'scale stddev';

done_testing();
