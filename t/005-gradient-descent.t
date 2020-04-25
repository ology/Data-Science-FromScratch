#!/usr/bin/env perl
use strict;
use warnings;

use Test::More;
use Statistics::Distribution::Generator qw(uniform);

use_ok 'Data::Science::FromScratch';

my $ds = new_ok 'Data::Science::FromScratch';

my $lower_slope     = 19.9;
my $upper_slope     = 20.1;
my $lower_intercept = 4.9;
my $upper_intercept = 5.1;

my $v = [ map { 0 + uniform(-10, 10) } 1 .. 3 ];
for my $i (1 .. 1000) {
    my $grad = $ds->doubled_gradient($v);
    $v = $ds->gradient_step($v, $grad, -0.01);
#    diag "$i. @$v";
}
ok $ds->distance($v, [0,0,0]) < 0.001, 'distance';

my @inputs;
push @inputs, [$_, 20 * $_ + 5]
    for -50 .. 49;

my $rate = 0.001;

$v = [ 0 + uniform(-1, 1), 0 + uniform(-1, 1)];
for my $i (1 .. 5000) {
    my @linear_gradients;
    for my $j (@inputs) {
        push @linear_gradients, $ds->linear_gradient(@$j, $v);
    }
    my $grad = $ds->vector_mean(@linear_gradients);
    $v = $ds->gradient_step($v, $grad, -$rate);
#    diag "$i. @$v";
}
my ($slope, $intercept) = @$v;
ok $lower_slope < $slope && $slope < $upper_slope, 'slope';
ok $lower_intercept < $intercept && $intercept < $upper_intercept, 'intercept';

$v = [ 0 + uniform(-1, 1), 0 + uniform(-1, 1)];
for my $i (1 .. 5000) {
    my @linear_gradients;
    my $iterator = $ds->minibatches(\@inputs, 20);
    while (my $batch = $iterator->next) {
        for my $x (@$batch) {
            push @linear_gradients, $ds->linear_gradient(@$x, $v);
        }
    }
    my $grad = $ds->vector_mean(@linear_gradients);
    $v = $ds->gradient_step($v, $grad, -$rate);
#    diag "$i. @$v";
}
($slope, $intercept) = @$v;
ok $lower_slope < $slope && $slope < $upper_slope, 'slope';
ok $lower_intercept < $intercept && $intercept < $upper_intercept, 'intercept';

$v = [ 0 + uniform(-1, 1), 0 + uniform(-1, 1)];
for my $i (1 .. 100) {
    for my $j (@inputs) {
        my $grad = $ds->linear_gradient(@$j, $v);
        $v = $ds->gradient_step($v, $grad, -$rate);
    }
#    diag "$i. @$v";
}
($slope, $intercept) = @$v;
ok $lower_slope < $slope && $slope < $upper_slope, 'slope';
ok $lower_intercept < $intercept && $intercept < $upper_intercept, 'intercept';

done_testing();
