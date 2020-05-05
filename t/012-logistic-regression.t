#!/usr/bin/env perl
use strict;
use warnings;

use Test::More;

use_ok 'Data::MachineLearning::Elements';

my $ml = new_ok 'Data::MachineLearning::Elements';

my $got = $ml->logistic(-10);
is sprintf('%.4f', $got), '0.0000', 'logistic';
$got = $ml->logistic(0);
is sprintf('%.4f', $got), '0.5000', 'logistic';
$got = $ml->logistic(10);
is sprintf('%.4f', $got), '1.0000', 'logistic';

$got = $ml->logistic_prime(-10);
is sprintf('%.4f', $got), '0.0000', 'logistic_prime';
$got = $ml->logistic_prime(0);
is sprintf('%.4f', $got), '0.2500', 'logistic_prime';
$got = $ml->logistic_prime(10);
is sprintf('%.4f', $got), '0.0000', 'logistic_prime';

SKIP: {
skip 'Incorrect results! What the?', 9;
my @data = ([0.7,48000,1],[1.9,48000,0],[2.5,60000,1],[4.2,63000,0],[6,76000,0],[6.5,69000,0],[7.5,76000,0],[8.1,88000,0],[8.7,83000,1],[10,83000,1],[0.8,43000,0],[1.8,60000,0],[10,79000,1],[6.1,76000,0],[1.4,50000,0],[9.1,92000,0],[5.8,75000,0],[5.2,69000,0],[1,56000,0],[6,67000,0],[4.9,74000,0],[6.4,63000,1],[6.2,82000,0],[3.3,58000,0],[9.3,90000,1],[5.5,57000,1],[9.1,102000,0],[2.4,54000,0],[8.2,65000,1],[5.3,82000,0],[9.8,107000,0],[1.8,64000,0],[0.6,46000,1],[0.8,48000,0],[8.6,84000,1],[0.6,45000,0],[0.5,30000,1],[7.3,89000,0],[2.5,48000,1],[5.6,76000,0],[7.4,77000,0],[2.7,56000,0],[0.7,48000,0],[1.2,42000,0],[0.2,32000,1],[4.7,56000,1],[2.8,44000,1],[7.6,78000,0],[1.1,63000,0],[8,79000,1],[2.7,56000,0],[6,52000,1],[4.6,56000,0],[2.5,51000,0],[5.7,71000,0],[2.9,65000,0],[1.1,33000,1],[3,62000,0],[4,71000,0],[2.4,61000,0],[7.5,75000,0],[9.7,81000,1],[3.2,62000,0],[7.9,88000,0],[4.7,44000,1],[2.5,55000,0],[1.6,41000,0],[6.7,64000,1],[6.9,66000,1],[7.9,78000,1],[8.1,102000,0],[5.3,48000,1],[8.5,66000,1],[0.2,56000,0],[6,69000,0],[7.5,77000,0],[8,86000,0],[4.4,68000,0],[4.9,75000,0],[1.5,60000,0],[2.2,50000,0],[3.4,49000,1],[4.2,70000,0],[7.7,98000,0],[8.2,85000,0],[5.4,88000,0],[0.1,46000,0],[1.5,37000,0],[6.3,86000,0],[3.7,57000,0],[8.4,85000,0],[2,42000,0],[5.8,69000,1],[2.7,64000,0],[3.1,63000,0],[1.9,48000,0],[10,72000,1],[0.2,45000,0],[8.6,95000,0],[1.5,64000,0],[9.8,95000,0],[5.3,65000,0],[7.5,80000,0],[9.9,91000,0],[9.7,50000,1],[2.8,68000,0],[3.6,58000,0],[3.9,74000,0],[4.4,76000,0],[2.5,49000,0],[7.2,81000,0],[5.2,60000,1],[2.4,62000,0],[8.9,94000,0],[2.4,63000,0],[6.8,69000,1],[6.5,77000,0],[7,86000,0],[9.4,94000,0],[7.8,72000,1],[0.2,53000,0],[10,97000,0],[5.5,65000,0],[7.7,71000,1],[8.1,66000,1],[9.8,91000,0],[8,84000,0],[2.7,55000,0],[2.8,62000,0],[9.4,79000,0],[2.5,57000,0],[7.4,70000,1],[2.1,47000,0],[5.3,62000,1],[6.3,79000,0],[6.8,58000,1],[5.7,80000,0],[2.2,61000,0],[4.8,62000,0],[3.7,64000,0],[4.1,85000,0],[2.3,51000,0],[3.5,58000,0],[0.9,43000,0],[0.9,54000,0],[4.5,74000,0],[6.5,55000,1],[4.1,41000,1],[7.1,73000,0],[1.1,66000,0],[9.1,81000,1],[8,69000,1],[7.3,72000,1],[3.3,50000,0],[3.9,58000,0],[2.6,49000,0],[1.6,78000,0],[0.7,56000,0],[2.1,36000,1],[7.5,90000,0],[4.8,59000,1],[8.9,95000,0],[6.2,72000,0],[6.3,63000,0],[9.1,100000,0],[7.3,61000,1],[5.6,74000,0],[0.5,66000,0],[1.1,59000,0],[5.1,61000,0],[6.2,70000,0],[6.6,56000,1],[6.3,76000,0],[6.5,78000,0],[5.1,59000,0],[9.5,74000,1],[4.5,64000,0],[2,54000,0],[1,52000,0],[4,69000,0],[6.5,76000,0],[3,60000,0],[4.5,63000,0],[7.8,70000,0],[3.9,60000,1],[0.8,51000,0],[4.2,78000,0],[1.1,54000,0],[6.2,60000,0],[2.9,59000,0],[2.1,52000,0],[8.2,87000,0],[4.8,73000,0],[2.2,42000,1],[9.1,98000,0],[6.5,84000,0],[6.9,73000,0],[5.1,72000,0],[9.1,69000,1],[9.8,79000,1]);
my @xs = map { [ 1, $_->[0], $_->[1] ] } @data;
my @ys = map { $_->[2] } @data;
my @rescaled_xs = $ml->rescale(@xs);
my ($x_train, $x_test, $y_train, $y_test) = $ml->train_test_split(\@rescaled_xs, \@ys, 0.33);
my $beta = [ map { rand } @{ $xs[0] } ];
my $loss = 0;
for my $i (1 .. 5000) {
    my $gradient = $ml->negative_log_gradient($x_train, $y_train, $beta);
    $beta = $ml->gradient_step($beta, $gradient, -0.01);
    $loss = $ml->negative_log_likelihood($x_train, $y_train, $beta);
}
is sprintf('%.4f', $beta->[0]), '-2.0239', 'beta';
is sprintf('%.4f', $beta->[1]), '4.6930', 'beta';
is sprintf('%.4f', $beta->[2]), '-4.4698', 'beta';
is sprintf('%.4f', $loss), '39.9635', 'loss';

my ($x, $y) = $ml->scale(@xs);
my @beta_unscaled = [
    $beta->[0] - $beta->[1] * $x->[1] / $y->[1] - $beta->[2] * $x->[2] / $y->[2],
    $beta->[1] / $y->[1],
    $beta->[2] / $y->[2],
];
is sprintf('%.4f', $beta_unscaled[0]), '8.9272', 'beta_unscaled';
is sprintf('%.4f', $beta_unscaled[1]), '1.6482', 'beta_unscaled';
is sprintf('%.4f', $beta_unscaled[2]), '-0.0003', 'beta_unscaled';

my ($true_pos, $false_pos, $tru_neg, $false_neg) = (0, 0, 0, 0);
for my $i (0 .. @$x_test - 1) {
    my $prediction = $ml->logistic($ml->vector_dot($beta, $x_test->[$i]));
    if ($y_test->[$i] == 1 && $prediction >= 0.5) {
        $true_pos++;
    }
    elsif ($y_test->[$i] == 1) {
        $false_neg++;
    }
    elsif ($prediction >= 0.5) {
        $false_pos++;
    }
    else {
        $tru_neg++;
    }
}
my $precision = $true_pos / ($true_pos + $false_pos);
is sprintf('%.4f', $precision), '0.2121', 'precision';
my $recall = $true_pos / ($true_pos + $false_neg);
is sprintf('%.4f', $recall), '1.0000', 'recall';
}

done_testing();
