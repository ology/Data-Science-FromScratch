#!/usr/bin/env perl
use strict;
use warnings;

use Test::More;

use_ok 'Data::Science::FromScratch';

my $ds = new_ok 'Data::Science::FromScratch';

my @x_data = (1,2,3,4,5,6);
my @y_data = map { 2 * $_ } @x_data;

my ($train, $test) = $ds->split_data(0.5, @x_data);
is @$train, 3, 'split_data';
is @$test, 3, 'split_data';

my ($x_train, $x_test, $y_train, $y_test) = $ds->train_test_split(0.5, \@x_data, \@y_data);
is @$x_train, 3, 'train_test_split';
is @$x_test, 3, 'train_test_split';
is @$y_train, 3, 'train_test_split';
is @$y_test, 3, 'train_test_split';

@x_data = map { rand } 1 .. 1000;
@y_data = map { 2 * $_ } @x_data;

($train, $test) = $ds->split_data(0.75, @x_data);
is @$train, 750, 'split_data';
is @$test, 250, 'split_data';

($x_train, $x_test, $y_train, $y_test) = $ds->train_test_split(0.25, \@x_data, \@y_data);
is @$x_train, 750, 'train_test_split';
is @$x_test, 250, 'train_test_split';
is @$y_train, 750, 'train_test_split';
is @$y_test, 250, 'train_test_split';

@x_data = (70, 4930, 13930, 981070);

my $got = $ds->accuracy(@x_data);
is sprintf('%.4f', $got), '0.9811', 'accuracy';

$got = $ds->precision(@x_data);
is sprintf('%.4f', $got), '0.0140', 'precision';

$got = $ds->recall(@x_data);
is sprintf('%.4f', $got), '0.0050', 'recall';

$got = $ds->f1_score(@x_data);
is sprintf('%.4f', $got), '0.0074', 'f1_score';

done_testing();
