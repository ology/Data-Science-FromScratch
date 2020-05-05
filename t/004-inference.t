#!/usr/bin/env perl
use strict;
use warnings;

use Test::More;

use_ok 'Data::MachineLearning::Elements';

my $ml = new_ok 'Data::MachineLearning::Elements';

is_deeply $ml->normal_approximation_to_binomial(100, 0.5),
    [50,5], 'normal_approximation_to_binomial';

my $got = $ml->normal_probability_below(-1, 0, 1);
is sprintf('%.4f', $got), '0.1589', 'normal_probability_below';
$got = $ml->normal_probability_above(-1, 0, 1);
is sprintf('%.4f', $got), '0.8411', 'normal_probability_above';
$got = $ml->normal_probability_between(-1, 0, 0, 1);
is sprintf('%.4f', $got), '0.3411', 'normal_probability_between';
$got = $ml->normal_probability_outside(-1, 0, 0, 1);
is sprintf('%.4f', $got), '0.6589', 'normal_probability_outside';

$got = $ml->normal_upper_bound(0.05, 0, 1);
is sprintf('%.4f', $got), '-1.6431', 'normal_upper_bound';
$got = $ml->normal_lower_bound(0.05, 0, 1);
is sprintf('%.4f', $got), '1.6431', 'normal_lower_bound';
$got = $ml->normal_two_sided_bounds(0.05, 0, 1);
is sprintf('%.4f', $got->[0]), '-0.0632', 'normal_two_sided_bounds';
is sprintf('%.4f', $got->[1]), '0.0632', 'normal_two_sided_bounds';

$got = $ml->two_sided_p_value(1, 0, 1);
is sprintf('%.4f', $got), '0.3178', 'two_sided_p_value';

done_testing();
