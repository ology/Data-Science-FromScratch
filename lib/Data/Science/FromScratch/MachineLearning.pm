package Data::Science::FromScratch::MachineLearning;

use List::Util qw(shuffle);
use Moo::Role;
use strictures 2;

=head1 SYNOPSIS

  use Data::Science::FromScratch;

  my $ds = Data::Science::FromScratch->new;

  my @x_data = (1,2,3,4,5,6);
  my @y_data = (2,4,6,8,10,12);

  my ($train, $test) = $ds->split_data(0.5, @x_data); # 3 element vectors

  my ($x_train, $x_test, $y_train, $y_test) = $ds->train_test_split(0.5, \@x_data, \@y_data);
    # 3 element vectors

  my $x = $ds->accuracy(70, 4930, 13930, 981070); # 0.9811
  my $x = $ds->precision(70, 4930, 13930, 981070); # 0.0140
  my $x = $ds->recall(70, 4930, 13930, 981070); # 0.0050
  my $x = $ds->f1_score(70, 4930, 13930, 981070); # 0.0074

=head1 METHODS

=head2 split_data

  ($train, $test) = $ds->split_data($probability, @data);

=cut

sub split_data {
    my ($self, $probability, @data) = @_;
    @data = shuffle @data;
    my $cut = int(@data * $probability);
    return [ @data[0 .. $cut - 1] ], [ @data[$cut .. @data - 1] ];
}

=head2 train_test_split

  ($x_train, $x_test, $y_train, $y_test) = $ds->train_test_split($probability, \@x_data, \@y_data);

=cut

sub train_test_split {
    my ($self, $probability, $x_data, $y_data) = @_;
    my @indices = (0 .. @$x_data - 1);
    my ($train_idx, $test_idx) = $self->split_data(1 - $probability, @indices);
    my @x_train = map { $x_data->[$_] } @$train_idx;
    my @x_test  = map { $x_data->[$_] } @$test_idx;
    my @y_train = map { $y_data->[$_] } @$train_idx;
    my @y_test  = map { $y_data->[$_] } @$test_idx;
    return \@x_train, \@x_test, \@y_train, \@y_test;
}

=head2 accuracy

  $x = $ds->accuracy($true_pos, $false_pos, $false_neg, $true_neg);

=cut

sub accuracy {
    my ($self, $true_pos, $false_pos, $false_neg, $true_neg) = @_;
    my $correct = $true_pos + $true_neg;
    my $total = $true_pos + $false_pos + $false_neg + $true_neg;
    return $correct / $total;
}

=head2 precision

  $x = $ds->precision($true_pos, $false_pos, $false_neg, $true_neg);

=cut

sub precision {
    my ($self, $true_pos, $false_pos, $false_neg, $true_neg) = @_;
    return $true_pos / ($true_pos + $false_pos);
}

=head2 recall

  $x = $ds->recall($true_pos, $false_pos, $false_neg, $true_neg);

=cut

sub recall {
    my ($self, $true_pos, $false_pos, $false_neg, $true_neg) = @_;
    return $true_pos / ($true_pos + $false_neg);
}

=head2 f1_score

  $x = $ds->f1_score($true_pos, $false_pos, $false_neg, $true_neg);

=cut

sub f1_score {
    my ($self, $true_pos, $false_pos, $false_neg, $true_neg) = @_;
    my $precision = $self->precision($true_pos, $false_pos, $false_neg, $true_neg);
    my $recall = $self->recall($true_pos, $false_pos, $false_neg, $true_neg);
    return 2 * $precision * $recall / ($precision + $recall);
}

1;
__END__

=head1 SEE ALSO

L<Data::Science::FromScratch>

F<t/006-machine-learning.t>

L<List::Util>

L<Moo::Role>

=cut
