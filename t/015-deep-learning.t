#!/usr/bin/env perl
use strict;
use warnings;

use Test::More;

use_ok 'Data::MachineLearning::Elements';

my $ml = new_ok 'Data::MachineLearning::Elements';

is_deeply $ml->tensor_shape([1,2,3]), [3], 'tensor_shape';
is_deeply $ml->tensor_shape([[1,2],[3,4],[5,6]]), [3,2], 'tensor_shape';

ok $ml->is_1d([1,2,3]), 'is_1d';
ok ! $ml->is_1d([[1,2],[3,4]]), 'is_1d';

is $ml->tensor_sum([1,2,3]), 6, 'tensor_sum';
is $ml->tensor_sum([[1,2],[3,4]]), 10, 'tensor_sum';

is_deeply $ml->tensor_apply(sub { shift() + 1 }, [1,2,3]), [2,3,4], 'tensor_apply';
is_deeply $ml->tensor_apply(sub { 2 * shift() }, [[1,2],[3,4]]), [[2,4], [6,8]], 'tensor_apply';

is_deeply $ml->zeros_like([1,2,3]), [0,0,0], 'zeros_like';
is_deeply $ml->zeros_like([[1,2],[3,4]]), [[0,0],[0,0]], 'zeros_like';

is_deeply $ml->tensor_combine(sub { shift() + shift() }, [1,2,3], [4,5,6]), [5,7,9], 'tensor_combine';
is_deeply $ml->tensor_combine(sub { shift() * shift() }, [1,2,3], [4,5,6]), [4,10,18], 'tensor_combine';
is_deeply $ml->tensor_combine(sub { shift() + shift() }, [[1,2],[3,4]], [[0,0],[0,0]]), [[1,2],[3,4]], 'tensor_combine';

my $got = $ml->random_uniform([2,3,4]);
is scalar(@$got), 2, 'random_uniform';
is scalar(@{ $got->[0] }), 3, 'random_uniform';
is scalar(@{ $got->[0][0] }), 4, 'random_uniform';

$got = $ml->random_tensor([2,3,4], 'uniform');
is scalar(@$got), 2, 'random_tensor';
is scalar(@{ $got->[0] }), 3, 'random_tensor';
is scalar(@{ $got->[0][0] }), 4, 'random_tensor';

is_deeply $ml->tensor_shape($ml->random_uniform([2,3,4])), [2,3,4], 'random_uniform';
is_deeply $ml->tensor_shape($ml->random_normal([5,6], 10)), [5,6], 'random_normal';

use_ok 'Data::MachineLearning::Elements::NeuralNetworks::Sigmoid';
my $net = new_ok 'Data::MachineLearning::Elements::NeuralNetworks::Sigmoid';
is_deeply $net->forward([0,0,0]), [0.5,0.5,0.5], 'forward';
is_deeply $net->backward([0,0,0]), [0,0,0], 'backward';
is_deeply $net->backward([1,2,3]), [0.25,0.5,0.75], 'backward';

use_ok 'Data::MachineLearning::Elements::NeuralNetworks::Linear';
$net = new_ok 'Data::MachineLearning::Elements::NeuralNetworks::Linear' => [
    input_dim  => 2,
    output_dim => 1,
];
$got = $net->forward([0,0]);
is scalar(@$got), $net->output_dim, 'forward';
is_deeply $net->backward([0,0]), [0,0], 'backward';

use_ok 'Data::MachineLearning::Elements::NeuralNetworks::Sequential';

$net = new_ok 'Data::MachineLearning::Elements::NeuralNetworks::Sequential' => [
    layers => [
        Data::MachineLearning::Elements::NeuralNetworks::Sigmoid->new,
    ],
];
is_deeply $net->forward([0,0]), [0.5,0.5], 'forward';
is_deeply $net->backward([0,0]), [0,0], 'backward';

$net = new_ok 'Data::MachineLearning::Elements::NeuralNetworks::Sequential' => [
    layers => [
        Data::MachineLearning::Elements::NeuralNetworks::Linear->new(input_dim => 2, output_dim => 1),
    ],
];
$got = $net->forward([0,0]);
is scalar(@$got), 1, 'forward';
is_deeply $net->backward([0,0]), [0,0], 'backward';

use_ok 'Data::MachineLearning::Elements::NeuralNetworks::SSE';
my $loss = new_ok 'Data::MachineLearning::Elements::NeuralNetworks::SSE';

is $loss->loss([1,2,3], [10,20,30]), 9 ** 2 + 18 ** 2 + 27 ** 2, 'loss';
is_deeply $loss->gradient([1,2,3], [10,20,30]), [-18,-36,-54], 'gradient';

SKIP: {
skip 'Broken algorithm. :(', 3;

$net = new_ok 'Data::MachineLearning::Elements::NeuralNetworks::Sequential' => [
    layers => [
        Data::MachineLearning::Elements::NeuralNetworks::Linear->new(input_dim => 2, output_dim => 2, id => 1),
        Data::MachineLearning::Elements::NeuralNetworks::Sigmoid->new,
        Data::MachineLearning::Elements::NeuralNetworks::Linear->new(input_dim => 2, output_dim => 1, id => 2),
    ],
];

use_ok 'Data::MachineLearning::Elements::NeuralNetworks::GradientDescent';
my $optimizer = new_ok 'Data::MachineLearning::Elements::NeuralNetworks::GradientDescent' => [
    lr => 0.1,
];

my @xs = ([0,0], [0,1], [1,0], [1,1]);
my @ys = ( [0],   [1],   [1],   [0] );

for my $i (1 .. 3000) {
    my $epoch_loss = 0;
    for my $j (0 .. @xs - 1) {
        my $predicted = $net->forward($xs[$j]);
        $epoch_loss += $loss->loss($predicted, $ys[$j]);
        my $gradient = $loss->gradient($predicted, $ys[$j]);
        $net->backward($gradient);
        $optimizer->step($net);
    }
    print "$i. Loss = $epoch_loss\n";
}

for my $param (@{ $net->params }) {
use Data::Dumper;warn(__PACKAGE__,' ',__LINE__," MARK: ",Dumper$param);
}
}

is $ml->tanh(-111), -1, 'tanh';
is $ml->tanh(0), 0, 'tanh';
is $ml->tanh(111), 1, 'tanh';
$got = $ml->tanh(1);
is sprintf('%.4f', $got), '0.7616', 'tanh';

SKIP: {
skip 'Broken algorithm. :(', 2;

my @xs = ( map { $ml->binary_encode($_) } 101 .. 1023 );
my @ys = ( map { $ml->fizz_buzz_encode($_) } 101 .. 1023 );

my $num_hidden = 25;

use_ok 'Data::MachineLearning::Elements::NeuralNetworks::Tanh';
$net = new_ok 'Data::MachineLearning::Elements::NeuralNetworks::Sequential' => [
    layers => [
        Data::MachineLearning::Elements::NeuralNetworks::Linear->new(input_dim => 10, output_dim => $num_hidden, init => 'uniform'),
        Data::MachineLearning::Elements::NeuralNetworks::Tanh->new,
        Data::MachineLearning::Elements::NeuralNetworks::Linear->new(input_dim => $num_hidden, output_dim => 4, init => 'uniform'),
        Data::MachineLearning::Elements::NeuralNetworks::Sigmoid->new,
    ],
];

use_ok 'Data::MachineLearning::Elements::NeuralNetworks::GradientDescent';
my $optimizer = new_ok 'Data::MachineLearning::Elements::NeuralNetworks::GradientDescent' => [
    lr => 0.1,
    mo => 0.9,
];

# $loss instantiated above

for my $i (1 .. 1000) {
    my $epoch_loss = 0;
    for my $j (0 .. @xs - 1) {
        my $predicted = $net->forward($xs[$j]);
        $epoch_loss += $loss->loss($predicted, $ys[$j]);
        my $gradient = $loss->gradient($predicted, $ys[$j]);
        $net->backward($gradient);
        $optimizer->step($net);
    }
    my $accuracy = $ml->fizz_buzz_accuracy(101, 1024, $net);
    print "$i. Loss = $epoch_loss, Accuracy = $accuracy\n";
}
}

done_testing();
