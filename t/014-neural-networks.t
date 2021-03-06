#!/usr/bin/env perl
use strict;
use warnings;

use Test::More;

use List::MoreUtils qw(zip6);

use_ok 'Data::MachineLearning::Elements';

my $ml = new_ok 'Data::MachineLearning::Elements';

is $ml->step_function(-1), 0, 'step_function';
is $ml->step_function(0), 1, 'step_function';
is $ml->step_function(1), 1, 'step_function';

my $weights = [2,2];
my $bias = -3;

# AND gate
is $ml->perceptron_output($weights, $bias, [1,1]), 1, 'perceptron_output';
is $ml->perceptron_output($weights, $bias, [0,1]), 0, 'perceptron_output';
is $ml->perceptron_output($weights, $bias, [1,0]), 0, 'perceptron_output';
is $ml->perceptron_output($weights, $bias, [0,0]), 0, 'perceptron_output';

$bias = -1;

# OR gate
is $ml->perceptron_output($weights, $bias, [1,1]), 1, 'perceptron_output';
is $ml->perceptron_output($weights, $bias, [0,1]), 1, 'perceptron_output';
is $ml->perceptron_output($weights, $bias, [1,0]), 1, 'perceptron_output';
is $ml->perceptron_output($weights, $bias, [0,0]), 0, 'perceptron_output';

$weights = [-2];
$bias = 1;

# NOT gate
is $ml->perceptron_output($weights, $bias, [0]), 1, 'perceptron_output';
is $ml->perceptron_output($weights, $bias, [1]), 0, 'perceptron_output';

my $got = $ml->sigmoid(-10);
is sprintf('%.4f', $got), '0.0000', 'sigmoid';
$got = $ml->sigmoid(-5);
is sprintf('%.4f', $got), '0.0067', 'sigmoid';
$got = $ml->sigmoid(0);
is sprintf('%.4f', $got), '0.5000', 'sigmoid';
$got = $ml->sigmoid(5);
is sprintf('%.4f', $got), '0.9933', 'sigmoid';
$got = $ml->sigmoid(10);
is sprintf('%.4f', $got), '1.0000', 'sigmoid';

my $net = [
    [ # Hidden layer
        [20,20,-30], # AND neuron
        [20,20,-10], # OR neuron
    ],
    [ # Output layer
        [-60,60,-30],
    ],
];

# XOR gate
$got = $ml->feed_forward($net, [0,0]);
ok $got->[-1][0] > 0 && $got->[-1][0] < 0.001, 'feed_forward';
$got = $ml->feed_forward($net, [1,0]);
ok $got->[-1][0] > 0.999 && $got->[-1][0] < 1, 'feed_forward';
$got = $ml->feed_forward($net, [0,1]);
ok $got->[-1][0] > 0.999 && $got->[-1][0] < 1, 'feed_forward';
$got = $ml->feed_forward($net, [1,1]);
ok $got->[-1][0] > 0 && $got->[-1][0] < 0.001, 'feed_forward';

SKIP: {
skip 'Long running test', 4;
# XXX This test passes and fails intermittently. :\
my $xs = [[0,0], [0,1], [1,0], [1,1]];
my $ys = [ [0],   [1],   [1],   [0]];

$net = [
    [
        [ rand, rand, rand ],
        [ rand, rand, rand ],
    ],
    [
        [ rand, rand, rand ],
    ],
];

for my $i (1 .. 20_000) {
    for my $j (0 .. @$xs - 1) {
        my $gradients = $ml->sqerror_gradients($net, $xs->[$j], $ys->[$j]);
        my @step;
        for my $k (0 .. @$net - 1) {
            my @x;
            my @zipped = zip6(@{ $net->[$k] }, @{ $gradients->[$k] });
            for my $z (@zipped) {
                push @x, $ml->gradient_step($z->[0], $z->[1], -1);
            }
            push @step, \@x;
        }
        $net = \@step;
    }
}

$got = $ml->feed_forward($net, [0,0]);
ok $got->[-1][0] < 0.01, 'feed_forward';
$got = $ml->feed_forward($net, [0,1]);
ok $got->[-1][0] > 0.99, 'feed_forward';
$got = $ml->feed_forward($net, [1,0]);
ok $got->[-1][0] > 0.99, 'feed_forward';
$got = $ml->feed_forward($net, [1,1]);
ok $got->[-1][0] < 0.01, 'feed_forward';
}

is_deeply $ml->fizz_buzz_encode(2), [1,0,0,0], 'fizz_buzz_encode';
is_deeply $ml->fizz_buzz_encode(6), [0,1,0,0], 'fizz_buzz_encode';
is_deeply $ml->fizz_buzz_encode(10), [0,0,1,0], 'fizz_buzz_encode';
is_deeply $ml->fizz_buzz_encode(30), [0,0,0,1], 'fizz_buzz_encode';

is_deeply $ml->binary_encode(0), [0,0,0,0,0,0,0,0,0,0], 'binary_encode';
is_deeply $ml->binary_encode(1), [1,0,0,0,0,0,0,0,0,0], 'binary_encode';
is_deeply $ml->binary_encode(10), [0,1,0,1,0,0,0,0,0,0], 'binary_encode';
is_deeply $ml->binary_encode(101), [1,0,1,0,0,1,1,0,0,0], 'binary_encode';
is_deeply $ml->binary_encode(999), [1,1,1,0,0,1,1,1,1,1], 'binary_encode';

is $ml->argmax([0,-1]), 0, 'argmax';
is $ml->argmax([-1,0]), 1, 'argmax';
is $ml->argmax([-1,10,5,20,-3]), 3, 'argmax';

SKIP: {
skip 'Long running test', 1;
my $xs = [ map { $ml->binary_encode($_) } 101 .. 1023 ];
my $ys = [ map { $ml->fizz_buzz_encode($_) } 101 .. 1023 ];

my $num_hidden = 25;
$net = [
    [ map { [ map { rand } 0 .. 10 ] } 1 .. $num_hidden ],
    [ map { [ map { rand } 0 .. $num_hidden ] } 1 .. 4 ],
];

for my $i (1 .. 500) {
    my $epoch_loss = 0;
    for my $j (0 .. @$xs - 1) {
        my $predicted = $ml->feed_forward($net, $xs->[$j])->[-1];
        $epoch_loss += $ml->squared_distance($predicted, $ys->[$j]);
        my $gradients = $ml->sqerror_gradients($net, $xs->[$j], $ys->[$j]);
        my @step;
        for my $k (0 .. @$net - 1) {
            my @x;
            my @zipped = zip6(@{ $net->[$k] }, @{ $gradients->[$k] });
            for my $z (@zipped) {
                push @x, $ml->gradient_step($z->[0], $z->[1], -1);
            }
            push @step, \@x;
        }
        $net = \@step;
    }
    print "$i. Loss = $epoch_loss\n";
}

my $num_correct = 0;
for my $i (1 .. 100) {
    my $x = $ml->binary_encode($i);
    my $predicted = $ml->argmax($ml->feed_forward($net, $x)->[-1]);
    my $actual = $ml->argmax($ml->fizz_buzz_encode($i));
    my @labels = ($i, 'fizz', 'buzz', 'fizzbuzz');
    printf "%d. %s %s\n", $i, $labels[$predicted], $labels[$actual];
    if ($predicted == $actual) {
        $num_correct++;
    }
}
ok $num_correct > 95, 'num_correct';
}

done_testing();
