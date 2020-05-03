#!/usr/bin/env perl
use strict;
use warnings;

use Test::More;

use_ok 'Data::Science::FromScratch';

my $ds = new_ok 'Data::Science::FromScratch';

is_deeply $ds->tensor_shape([1,2,3]), [3], 'tensor_shape';
is_deeply $ds->tensor_shape([[1,2],[3,4],[5,6]]), [3,2], 'tensor_shape';

ok $ds->is_1d([1,2,3]), 'is_1d';
ok ! $ds->is_1d([[1,2],[3,4]]), 'is_1d';

is $ds->tensor_sum([1,2,3]), 6, 'tensor_sum';
is $ds->tensor_sum([[1,2],[3,4]]), 10, 'tensor_sum';

is_deeply $ds->tensor_apply(sub { shift() + 1 }, [1,2,3]), [2,3,4], 'tensor_apply';
is_deeply $ds->tensor_apply(sub { 2 * shift() }, [[1,2],[3,4]]), [[2,4], [6,8]], 'tensor_apply';

is_deeply $ds->zeros_like([1,2,3]), [0,0,0], 'zeros_like';
is_deeply $ds->zeros_like([[1,2],[3,4]]), [[0,0],[0,0]], 'zeros_like';

is_deeply $ds->tensor_combine(sub { shift() + shift() }, [1,2,3], [4,5,6]), [5,7,9], 'tensor_combine';
is_deeply $ds->tensor_combine(sub { shift() * shift() }, [1,2,3], [4,5,6]), [4,10,18], 'tensor_combine';

is_deeply $ds->tensor_shape($ds->random_uniform([2,3,4])), [2,3,4], 'random_uniform';
is_deeply $ds->tensor_shape($ds->random_normal([5,6], 10)), [5,6], 'random_normal';

use_ok 'Data::Science::FromScratch::NeuralNetworks::Sigmoid';
my $net = new_ok 'Data::Science::FromScratch::NeuralNetworks::Sigmoid';
is_deeply $net->forward([0,0,0]), [0.5,0.5,0.5], 'forward';
is_deeply $net->backward([0,0,0]), [0,0,0], 'backward';
is_deeply $net->backward([1,2,3]), [0.25,0.5,0.75], 'backward';

use_ok 'Data::Science::FromScratch::NeuralNetworks::Linear';
$net = new_ok 'Data::Science::FromScratch::NeuralNetworks::Linear' => [
    input_dim  => 2,
    output_dim => 1,
];
my $got = $net->forward([0,0]);
is scalar(@$got), $net->output_dim, 'forward';
is_deeply $net->backward([0,0]), [[0], [0]], 'backward';

use_ok 'Data::Science::FromScratch::NeuralNetworks::Sequential';

$net = new_ok 'Data::Science::FromScratch::NeuralNetworks::Sequential' => [
    layers => [
        Data::Science::FromScratch::NeuralNetworks::Sigmoid->new,
    ],
];
is_deeply $net->forward([0,0]), [0.5,0.5], 'forward';
is_deeply $net->backward([0,0]), [0,0], 'backward';

$net = new_ok 'Data::Science::FromScratch::NeuralNetworks::Sequential' => [
    layers => [
        Data::Science::FromScratch::NeuralNetworks::Linear->new(input_dim => 2, output_dim => 1),
    ],
];
$got = $net->forward([0,0]);
is scalar(@$got), 1, 'forward';
is_deeply $net->backward([0,0]), [[0], [0]], 'backward';

$net = new_ok 'Data::Science::FromScratch::NeuralNetworks::Sequential' => [
    layers => [
        Data::Science::FromScratch::NeuralNetworks::Linear->new(input_dim => 2, output_dim => 2),
        Data::Science::FromScratch::NeuralNetworks::Sigmoid->new,
        Data::Science::FromScratch::NeuralNetworks::Linear->new(input_dim => 2, output_dim => 1),
    ],
];

use_ok 'Data::Science::FromScratch::NeuralNetworks::GradientDescent';
my $optimizer = new_ok 'Data::Science::FromScratch::NeuralNetworks::GradientDescent' => [
    lr => 0.1,
];

use_ok 'Data::Science::FromScratch::NeuralNetworks::SSE';
my $loss = new_ok 'Data::Science::FromScratch::NeuralNetworks::SSE';

is $loss->loss([1,2,3], [10,20,30]), 9 ** 2 + 18 ** 2 + 27 ** 2, 'loss';
is_deeply $loss->gradient([1,2,3], [10,20,30]), [-18,-36,-54], 'gradient';

SKIP: {
skip 'Incorrect results! What the?', 0;

my @xs = ([0,0], [0,1], [1,0], [1,1]);
my @ys = ( [0],   [1],   [1],   [0]);

for my $i (1 .. 3000) {
    my $epoch_loss = 0;
    for my $j (0 .. @xs - 1) {
        my $predicted = $net->forward($xs[$j]);
        $epoch_loss += $loss->loss($predicted, $ys[$j]);
#warn(__PACKAGE__,' ',__LINE__," MARK: ",$epoch_loss,"\n");
        my $gradient = $loss->gradient($predicted, $ys[$j]);
        $net->backward($gradient);
        $optimizer->step($net);
    }
}

for my $param (@{ $net->params }) {
use Data::Dumper;warn(__PACKAGE__,' ',__LINE__," MARK: ",Dumper$param);
}
}

done_testing();
