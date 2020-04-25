#!/usr/bin/env perl
use strict;
use warnings;

use Test::More;

use_ok 'Data::Science::FromScratch';

my $ds = new_ok 'Data::Science::FromScratch';

is_deeply $ds->normal_approximation_to_binomial(100, 0.5),
    [50,5], 'normal_approximation_to_binomial';

my $got = $ds->normal_probability_below(-1, 0, 1);
is sprintf('%.4f', $got), '0.1589', 'normal_probability_below';
$got = $ds->normal_probability_above(-1, 0, 1);
is sprintf('%.4f', $got), '0.8411', 'normal_probability_above';
$got = $ds->normal_probability_between(-1, 0, 0, 1);
is sprintf('%.4f', $got), '0.3411', 'normal_probability_between';
$got = $ds->normal_probability_outside(-1, 0, 0, 1);
is sprintf('%.4f', $got), '0.6589', 'normal_probability_outside';

$got = $ds->normal_upper_bound(0.05, 0, 1);
is sprintf('%.4f', $got), '-1.6431', 'normal_upper_bound';
$got = $ds->normal_lower_bound(0.05, 0, 1);
is sprintf('%.4f', $got), '1.6431', 'normal_lower_bound';
$got = $ds->normal_two_sided_bounds(0.05, 0, 1);
is sprintf('%.4f', $got->[0]), '-0.0632', 'normal_two_sided_bounds';
is sprintf('%.4f', $got->[1]), '0.0632', 'normal_two_sided_bounds';

$got = $ds->two_sided_p_value(1, 0, 1);
is sprintf('%.4f', $got), '0.3178', 'two_sided_p_value';

done_testing();
