#!/usr/bin/env perl
use strict;
use warnings;

use Test::More;

use_ok 'Data::MachineLearning::Elements';

my $ml = new_ok 'Data::MachineLearning::Elements';

my @data = ([-3, -1, 1], [-1, 0, 1], [1, 1, 1]);

my ($x, $y) = $ml->scale(@data);
is_deeply $x, [-1, 0, 1], 'scale means';
is_deeply $y, [2, 1, 0], 'scale stddev';

($x, $y) = $ml->scale($ml->rescale(@data));
is_deeply $x, [0, 0, 1], 'scale means';
is_deeply $y, [1, 1, 0], 'scale stddev';

done_testing();
