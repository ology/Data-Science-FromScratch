package Data::MachineLearning::Elements::MachineLearning;

use List::Util qw(shuffle);
use Moo::Role;
use strictures 2;

=head1 SYNOPSIS

  use Data::MachineLearning::Elements;

  my $ds = Data::MachineLearning::Elements->new;

  my @x_data = (1,2,3,4,5,6);
  my @y_data = (2,4,6,8,10,12);

  my ($train, $test) = $ds->split_data(0.5, @x_data); # 3-element vectors

  my ($x_train, $x_test, $y_train, $y_test) = $ds->train_test_split(\@x_data, \@y_data, 0.5);
    # 3-element vectors

  my $y = $ds->accuracy(70, 4930, 13930, 981070); # 0.9811
  $y = $ds->precision(70, 4930, 13930, 981070); # 0.0140
  $y = $ds->recall(70, 4930, 13930, 981070); # 0.0050
  $y = $ds->f1_score(70, 4930, 13930, 981070); # 0.0074

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

  ($x_train, $x_test, $y_train, $y_test) = $ds->train_test_split(\@x_data, \@y_data, $probability);

=cut

sub train_test_split {
    my ($self, $x_data, $y_data, $probability) = @_;
    my @indices = (0 .. @$x_data - 1);
    my ($train_idx, $test_idx) = $self->split_data(1 - $probability, @indices);
    my @x_train = map { $x_data->[$_] } @$train_idx;
    my @x_test  = map { $x_data->[$_] } @$test_idx;
    my @y_train = map { $y_data->[$_] } @$train_idx;
    my @y_test  = map { $y_data->[$_] } @$test_idx;
    return \@x_train, \@x_test, \@y_train, \@y_test;
}

=head2 accuracy

  $y = $ds->accuracy($true_pos, $false_pos, $false_neg, $true_neg);

=cut

sub accuracy {
    my ($self, $true_pos, $false_pos, $false_neg, $true_neg) = @_;
    my $correct = $true_pos + $true_neg;
    my $total = $true_pos + $false_pos + $false_neg + $true_neg;
    return $correct / $total;
}

=head2 precision

  $y = $ds->precision($true_pos, $false_pos, $false_neg, $true_neg);

=cut

sub precision {
    my ($self, $true_pos, $false_pos, $false_neg, $true_neg) = @_;
    return $true_pos / ($true_pos + $false_pos);
}

=head2 recall

  $y = $ds->recall($true_pos, $false_pos, $false_neg, $true_neg);

=cut

sub recall {
    my ($self, $true_pos, $false_pos, $false_neg, $true_neg) = @_;
    return $true_pos / ($true_pos + $false_neg);
}

=head2 f1_score

  $y = $ds->f1_score($true_pos, $false_pos, $false_neg, $true_neg);

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

L<Data::MachineLearning::Elements>

F<t/007-machine-learning.t>

L<List::Util>

L<Moo::Role>

=cut
