#!/usr/bin/env perl
use strict;
use warnings;

use Test::More;

use_ok 'Data::MachineLearning::Elements';

my $ml = new_ok 'Data::MachineLearning::Elements';

is $ml->uniform_cdf(-1), 0, 'uniform_cdf';
is $ml->uniform_cdf(0), 0, 'uniform_cdf';
is $ml->uniform_cdf(0.5), 0.5, 'uniform_cdf';
is $ml->uniform_cdf(1), 1, 'uniform_cdf';
is $ml->uniform_cdf(1.5), 1, 'uniform_cdf';

my $got = $ml->normal_pdf(-2, 0, 1);
is sprintf('%.4f', $got), '0.0540', 'normal_pdf';
$got = $ml->normal_pdf(-1, 0, 1);
is sprintf('%.4f', $got), '0.2420', 'normal_pdf';
$got = $ml->normal_pdf(0, 0, 1);
is sprintf('%.4f', $got), '0.3989', 'normal_pdf';
$got = $ml->normal_pdf(1, 0, 1);
is sprintf('%.4f', $got), '0.2420', 'normal_pdf';
$got = $ml->normal_pdf(2, 0, 1);
is sprintf('%.4f', $got), '0.0540', 'normal_pdf';

$got = $ml->normal_cdf(-2, 0, 1);
is sprintf('%.4f', $got), '0.0226', 'normal_cdf';
$got = $ml->normal_cdf(-1, 0, 1);
is sprintf('%.4f', $got), '0.1589', 'normal_cdf';
$got = $ml->normal_cdf(0, 0, 1);
is $got, 0.5, 'normal_cdf';
$got = $ml->normal_cdf(1, 0, 1);
is sprintf('%.4f', $got), '0.8411', 'normal_cdf';
$got = $ml->normal_cdf(2, 0, 1);
is sprintf('%.4f', $got), '0.9774', 'normal_cdf';

#$got = inverse_normal_cdf($n, $mu, $sigma, $tolerance); # TODO

#$got = $ml->bernouli_trial(0.5);
can_ok $ml, 'bernouli_trial';

#$got = $ml->binomial(100, 0.5);
can_ok $ml, 'binomial';

done_testing();
