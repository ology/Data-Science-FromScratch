#!/usr/bin/env perl
use strict;
use warnings;

use Test::More;

use List::SomeUtils qw(zip);

use_ok 'Data::Science::FromScratch';

my $ds = new_ok 'Data::Science::FromScratch';

my @data1 = (1,2,3);
my @data2 = (4,4,4);
my $y = 30;

my $got = $ds->mr_predict(\@data1, \@data2);
is $got, 24, 'mr_predict';

$got = $ds->mr_error(\@data1, $y, \@data2);
is $got, -6, 'mr_error';

$got = $ds->squared_error(\@data1, $y, \@data2);
is $got, 36, 'squared_error';

$got = $ds->sqerror_gradient(\@data1, $y, \@data2);
is_deeply $got, [-12, -24, -36], 'sqerror_gradient';

$got = $ds->bootstrap_sample([1,2,3,4,5]);
isa_ok $got, 'ARRAY';
is scalar @$got, 5, 'bootstrap_sample';

$got = $ds->bootstrap_statistic([1,2,3,4,5], 100);
isa_ok $got, 'ARRAY';
is scalar @$got, 100, 'bootstrap_statistic';

@data1 = map { 99.5 + rand } 1 .. 101;
@data2 = (
    (99.5 + rand),
    (map { rand } 1 .. 50),
    (map { 200 + rand } 1 .. 50),
);
$got = $ds->bootstrap_statistic(\@data1, 100);
isa_ok $got, 'ARRAY';
$got = $ds->standard_deviation(@$got);
ok $got < 1, 'bootstrap_statistic';
$got = $ds->bootstrap_statistic(\@data2, 100);
isa_ok $got, 'ARRAY';
$got = $ds->standard_deviation(@$got);
ok $got > 90, 'bootstrap_statistic';

$got = $ds->p_value(30.58, 1.27);
ok $got < 0.001, 'p_value';
$got = $ds->p_value(0.972, 0.103);
ok $got < 0.001, 'p_value';
$got = $ds->p_value(-1.865, 0.155);
ok $got < 0.001, 'p_value';
$got = $ds->p_value(0.923, 1.249);
ok $got > 0.4, 'p_value';

$got = $ds->ridge_penalty([1,2,3,4,5], 0.5);
is $got, 27, 'ridge_penalty';

$got = $ds->lasso_penalty([1,2,3,4,5], 0.5);
is $got, 7, 'lasso_penalty';

done_testing();
