#!/usr/bin/env perl
use strict;
use warnings;

use Test::More;

use_ok 'Data::MachineLearning::Elements';

my $ml = new_ok 'Data::MachineLearning::Elements';

my @data = ([-3, -1, 1], [-1, 0, 1], [1, 1, 1]);

my ($x, $y) = $ml->scale(@data);
is_deeply $x, [-1, 0, 1], 'scale means';
is_deeply $y, [2, 1, 0], 'scale stddev';

($x, $y) = $ml->scale($ml->rescale(@data));
is_deeply $x, [0, 0, 1], 'rescaled means';
is_deeply $y, [1, 1, 0], 'rescaled stddev';

@data = ([1, 0.7, 48000], [1, 1.9, 48000], [1, 2.5, 60000], [1, 4.2, 63000], [1, 6, 76000], [1, 6.5, 69000], [1, 7.5, 76000], [1, 8.1, 88000], [1, 8.7, 83000], [1, 10, 83000], [1, 0.8, 43000], [1, 1.8, 60000], [1, 10, 79000], [1, 6.1, 76000], [1, 1.4, 50000], [1, 9.1, 92000], [1, 5.8, 75000], [1, 5.2, 69000], [1, 1, 56000], [1, 6, 67000], [1, 4.9, 74000], [1, 6.4, 63000], [1, 6.2, 82000], [1, 3.3, 58000], [1, 9.3, 90000], [1, 5.5, 57000], [1, 9.1, 102000], [1, 2.4, 54000], [1, 8.2, 65000], [1, 5.3, 82000], [1, 9.8, 107000], [1, 1.8, 64000], [1, 0.6, 46000], [1, 0.8, 48000], [1, 8.6, 84000], [1, 0.6, 45000], [1, 0.5, 30000], [1, 7.3, 89000], [1, 2.5, 48000], [1, 5.6, 76000], [1, 7.4, 77000], [1, 2.7, 56000], [1, 0.7, 48000], [1, 1.2, 42000], [1, 0.2, 32000], [1, 4.7, 56000], [1, 2.8, 44000], [1, 7.6, 78000], [1, 1.1, 63000], [1, 8, 79000], [1, 2.7, 56000], [1, 6, 52000], [1, 4.6, 56000], [1, 2.5, 51000], [1, 5.7, 71000], [1, 2.9, 65000], [1, 1.1, 33000], [1, 3, 62000], [1, 4, 71000], [1, 2.4, 61000], [1, 7.5, 75000], [1, 9.7, 81000], [1, 3.2, 62000], [1, 7.9, 88000], [1, 4.7, 44000], [1, 2.5, 55000], [1, 1.6, 41000], [1, 6.7, 64000], [1, 6.9, 66000], [1, 7.9, 78000], [1, 8.1, 102000], [1, 5.3, 48000], [1, 8.5, 66000], [1, 0.2, 56000], [1, 6, 69000], [1, 7.5, 77000], [1, 8, 86000], [1, 4.4, 68000], [1, 4.9, 75000], [1, 1.5, 60000], [1, 2.2, 50000], [1, 3.4, 49000], [1, 4.2, 70000], [1, 7.7, 98000], [1, 8.2, 85000], [1, 5.4, 88000], [1, 0.1, 46000], [1, 1.5, 37000], [1, 6.3, 86000], [1, 3.7, 57000], [1, 8.4, 85000], [1, 2, 42000], [1, 5.8, 69000], [1, 2.7, 64000], [1, 3.1, 63000], [1, 1.9, 48000], [1, 10, 72000], [1, 0.2, 45000], [1, 8.6, 95000], [1, 1.5, 64000], [1, 9.8, 95000], [1, 5.3, 65000], [1, 7.5, 80000], [1, 9.9, 91000], [1, 9.7, 50000], [1, 2.8, 68000], [1, 3.6, 58000], [1, 3.9, 74000], [1, 4.4, 76000], [1, 2.5, 49000], [1, 7.2, 81000], [1, 5.2, 60000], [1, 2.4, 62000], [1, 8.9, 94000], [1, 2.4, 63000], [1, 6.8, 69000], [1, 6.5, 77000], [1, 7, 86000], [1, 9.4, 94000], [1, 7.8, 72000], [1, 0.2, 53000], [1, 10, 97000], [1, 5.5, 65000], [1, 7.7, 71000], [1, 8.1, 66000], [1, 9.8, 91000], [1, 8, 84000], [1, 2.7, 55000], [1, 2.8, 62000], [1, 9.4, 79000], [1, 2.5, 57000], [1, 7.4, 70000], [1, 2.1, 47000], [1, 5.3, 62000], [1, 6.3, 79000], [1, 6.8, 58000], [1, 5.7, 80000], [1, 2.2, 61000], [1, 4.8, 62000], [1, 3.7, 64000], [1, 4.1, 85000], [1, 2.3, 51000], [1, 3.5, 58000], [1, 0.9, 43000], [1, 0.9, 54000], [1, 4.5, 74000], [1, 6.5, 55000], [1, 4.1, 41000], [1, 7.1, 73000], [1, 1.1, 66000], [1, 9.1, 81000], [1, 8, 69000], [1, 7.3, 72000], [1, 3.3, 50000], [1, 3.9, 58000], [1, 2.6, 49000], [1, 1.6, 78000], [1, 0.7, 56000], [1, 2.1, 36000], [1, 7.5, 90000], [1, 4.8, 59000], [1, 8.9, 95000], [1, 6.2, 72000], [1, 6.3, 63000], [1, 9.1, 100000], [1, 7.3, 61000], [1, 5.6, 74000], [1, 0.5, 66000], [1, 1.1, 59000], [1, 5.1, 61000], [1, 6.2, 70000], [1, 6.6, 56000], [1, 6.3, 76000], [1, 6.5, 78000], [1, 5.1, 59000], [1, 9.5, 74000], [1, 4.5, 64000], [1, 2, 54000], [1, 1, 52000], [1, 4, 69000], [1, 6.5, 76000], [1, 3, 60000], [1, 4.5, 63000], [1, 7.8, 70000], [1, 3.9, 60000], [1, 0.8, 51000], [1, 4.2, 78000], [1, 1.1, 54000], [1, 6.2, 60000], [1, 2.9, 59000], [1, 2.1, 52000], [1, 8.2, 87000], [1, 4.8, 73000], [1, 2.2, 42000], [1, 9.1, 98000], [1, 6.5, 84000], [1, 6.9, 73000], [1, 5.1, 72000], [1, 9.1, 69000], [1, 9.8, 79000]);

($x, $y) = $ml->scale(@data);
is $x->[0], 1, 'scale means';
is sprintf('%.4f', $x->[1]), '4.9980', 'scale means';
is $x->[2], 66700, 'scale means';
is $y->[0], 0, 'scale stddev';
is sprintf('%.4f', $y->[1]), '2.8474', 'scale stddev';
is sprintf('%.0f', $y->[2]), 15537, 'scale stddev';

my @data2 = $ml->rescale(@data);
is $data2[0][0], 1, 'rescale';
is sprintf('%.4f', $data2[0][1]), '-1.5095', 'rescale';
is sprintf('%.4f', $data2[0][2]), '-1.2036', 'rescale';

my $got = $ml->vector_de_mean([1,2], [3,4], [5,6]);
is_deeply $got, [[-2,-2], [0,0], [2,2]], 'vector_de_mean';

done_testing();
