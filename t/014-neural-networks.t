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

my $weights = [2,2];
my $bias = -3;

# AND gate
is $ds->perceptron_output($weights, $bias, [1,1]), 1, 'perceptron_output';
is $ds->perceptron_output($weights, $bias, [0,1]), 0, 'perceptron_output';
is $ds->perceptron_output($weights, $bias, [1,0]), 0, 'perceptron_output';
is $ds->perceptron_output($weights, $bias, [0,0]), 0, 'perceptron_output';

$bias = -1;

# OR gate
is $ds->perceptron_output($weights, $bias, [1,1]), 1, 'perceptron_output';
is $ds->perceptron_output($weights, $bias, [0,1]), 1, 'perceptron_output';
is $ds->perceptron_output($weights, $bias, [1,0]), 1, 'perceptron_output';
is $ds->perceptron_output($weights, $bias, [0,0]), 0, 'perceptron_output';

$weights = [-2];
$bias = 1;

# NOT gate
is $ds->perceptron_output($weights, $bias, [0]), 1, 'perceptron_output';
is $ds->perceptron_output($weights, $bias, [1]), 0, 'perceptron_output';

my $got = $ds->sigmoid(-10);
is sprintf('%.4f', $got), '0.0000', 'sigmoid';
$got = $ds->sigmoid(-5);
is sprintf('%.4f', $got), '0.0067', 'sigmoid';
$got = $ds->sigmoid(0);
is sprintf('%.4f', $got), '0.5000', 'sigmoid';
$got = $ds->sigmoid(5);
is sprintf('%.4f', $got), '0.9933', 'sigmoid';
$got = $ds->sigmoid(10);
is sprintf('%.4f', $got), '1.0000', 'sigmoid';

done_testing();
