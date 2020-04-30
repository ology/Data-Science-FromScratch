#!/usr/bin/env perl
use strict;
use warnings;

use Test::More;

use Data::Dataset::Classic::Iris;

use_ok 'Data::Science::FromScratch';

my $ds = new_ok 'Data::Science::FromScratch';

my @labels = qw(a b c b a);

is $ds->majority_vote(@labels), 'b', 'majority_vote';

# Create [ { point => [sepal_length, sepal_width, petal_length, petal_width], label => species } ] array rows
my $iris = Data::Dataset::Classic::Iris::get();
my @iris_data;
for my $i (0 .. @{ $iris->{species} } - 1) {
    push @iris_data, {
        label => $iris->{species}[$i],
        point => [
            $iris->{sepal_length}[$i], $iris->{sepal_width}[$i],
            $iris->{petal_length}[$i], $iris->{petal_width}[$i],
        ],
    };
}

my ($train, $test) = $ds->split_data(0.70, @iris_data);
is @$train, 0.7 * 150, 'split_data';
is @$test, 0.3 * 150, 'split_data';

my %confusion_matrix;
my $num_correct = 0;
for my $i (@$test) {
    my $predicted = $ds->knn_classify(5, $train, $i->{point});
    my $actual = $i->{label};
    $num_correct++
        if $predicted eq $actual;
    $confusion_matrix{"$predicted,$actual"}++;
}
my $got = $num_correct / @$test;
ok 0.90 < $got, 'percent correct';
ok scalar(keys %confusion_matrix) > 3, 'confusion_matrix';

done_testing();
