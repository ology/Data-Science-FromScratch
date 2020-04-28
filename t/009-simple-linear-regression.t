#!/usr/bin/env perl
use strict;
use warnings;

use Test::More;

use List::Util qw(sum);

use_ok 'Data::Science::FromScratch';

my $ds = new_ok 'Data::Science::FromScratch';

my ($x, $y) = (0.5, 0.5);
my @data1 = (0.1, 0.5, 0.8);
my @data2 = (0.2, 0.4, 0.7);

my $got = $ds->lr_predict($x, $y, 0.5);
is $got, 0.75, 'lr_predict';

$got = $ds->lr_error($x, $y, 0.5, 0.4);
is $got, 0.35, 'lr_error';

$got = $ds->sum_of_sqerrors($x, $y, \@data1, \@data2);
is $got, 0.285, 'sum_of_sqerrors';

$got = $ds->total_sum_of_squares([1 .. 10]);
is $got, 82.5, 'total_sum_of_squares';

$got = $ds->r_squared($x, $y, \@data1, \@data2);
is $got, -1.25, 'r_squared';

($x, $y) = $ds->least_squares_fit(\@data1, \@data2);
is sprintf('%.4f', $x), '0.1054', 'least_squares_fit';
is sprintf('%.4f', $y), '0.7027', 'least_squares_fit';

@data1 = ();
for (my $i = -100; $i <= 110; $i += 10) {
    push @data1, $i;
}
@data2 = map { 3 * $_ - 5 } @data1;

($x, $y) = $ds->least_squares_fit(\@data1, \@data2);
is sprintf('%.4f', $x), '-5.0000', 'least_squares_fit';
is sprintf('%.4f', $y), '3.0000', 'least_squares_fit';

# num_friends_good from the book
@data1 = (49,41,40,25,21,21,19,19,18,18,16,15,15,15,15,14,14,13,13,13,13,12,12,11,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,8,8,8,8,8,8,8,8,8,8,8,8,8,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1);
# daily_minutes_good from the book
@data2 = (68.77,51.25,52.08,38.36,44.54,57.13,51.4,41.42,31.22,34.76,54.01,38.79,47.59,49.1,27.66,41.03,36.73,48.65,28.12,46.62,35.57,32.98,35,26.07,23.77,39.73,40.57,31.65,31.21,36.32,20.45,21.93,26.02,27.34,23.49,46.94,30.5,33.8,24.23,21.4,27.94,32.24,40.57,25.07,19.42,22.39,18.42,46.96,23.72,26.41,26.97,36.76,40.32,35.02,29.47,30.2,31,38.11,38.18,36.31,21.03,30.86,36.07,28.66,29.08,37.28,15.28,24.17,22.31,30.17,25.53,19.85,35.37,44.6,17.23,13.47,26.33,35.02,32.09,24.81,19.33,28.77,24.26,31.98,25.73,24.86,16.28,34.51,15.23,39.72,40.8,26.06,35.76,34.76,16.13,44.04,18.03,19.65,32.62,35.59,39.43,14.18,35.24,40.13,41.82,35.45,36.07,43.67,24.61,20.9,21.9,18.79,27.61,27.21,26.61,29.77,20.59,27.53,13.82,33.2,25,33.1,36.65,18.63,14.87,22.2,36.81,25.53,24.62,26.25,18.21,28.08,19.42,29.79,32.8,35.99,28.32,27.79,35.88,29.06,36.28,14.1,36.63,37.49,26.9,18.58,38.48,24.48,18.95,33.55,14.24,29.04,32.51,25.63,22.22,19,32.73,15.16,13.9,27.2,32.01,29.27,33,13.74,20.42,27.32,18.23,35.35,28.48,9.08,24.62,20.12,35.26,19.92,31.02,16.49,12.16,30.7,31.22,34.65,13.13,27.51,33.2,31.57,14.1,33.42,17.44,10.12,24.42,9.82,23.39,30.93,15.03,21.67,31.09,33.29,22.61,26.89,23.48,8.38,27.81,32.35,23.84);

($x, $y) = $ds->least_squares_fit(\@data1, \@data2);
is sprintf('%.4f', $x), '22.9476', 'least_squares_fit';
is sprintf('%.4f', $y), '0.9039', 'least_squares_fit';

$got = $ds->r_squared($x, $y, \@data1, \@data2);
is sprintf('%.4f', $got), '0.3291', 'r_squared';

my $epochs = 10000;
my $rate = 0.00001;
my $guess = [rand, rand];

for my $t (0 .. $epochs) {
    ($x, $y) = @$guess;
    my @loss = map { 2 * $ds->lr_error($x, $y, $data1[$_], $data2[$_]) } 0 .. @data1 - 1;
    my $grad_a = sum(@loss);
    @loss = map { 2 * $ds->lr_error($x, $y, $data1[$_], $data2[$_]) * $data1[$_] } 0 .. @data1 - 1;
    my $grad_b = sum(@loss);
#    my $loss = $ds->sum_of_sqerrors($x, $y, \@data1, \@data2);
    $guess = $ds->gradient_step($guess, [$grad_a, $grad_b], -$rate);
}

($x, $y) = @$guess;
is sprintf('%.4f', $x), '22.9476', 'gradient descent';
is sprintf('%.4f', $y), '0.9039', 'gradient descent';

done_testing();
