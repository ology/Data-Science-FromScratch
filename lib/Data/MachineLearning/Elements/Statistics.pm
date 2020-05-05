package Data::MachineLearning::Elements::Statistics;

use List::Util qw(max min sum0);
use Moo::Role;
use strictures 2;

=head1 SYNOPSIS

  use Data::MachineLearning::Elements;

  my $ml = Data::MachineLearning::Elements->new;

  my $y = $ml->mean(1,2,3); # 6 / 3

  $y = $ml->median(1,10,2,9,5); # 5
  $y = $ml->median(1,9,2,10); # (2 + 9) / 2

  my @data = (1,1, 2,2,2, 3,3,3,3, 4,4,4,4,4, 5,5,5,5,5,5, 6,6,6,6,6,6);
  my @data2 = (4,4, 5,5,5, 6,6,6,6, 7,7,7,7,7, 8,8,8,8,8,8, 9,9,9,9,9,9);

  $y = $ml->quantile(0.10, @data); # 2
  $y = $ml->quantile(0.25, @data); # 3
  $y = $ml->quantile(0.50, @data); # 4 (median)
  $y = $ml->quantile(0.75, @data); # 5
  $y = $ml->quantile(0.90, @data); # 6

  my $v = $ml->mode(@data); # [5,6]

  $y = $ml->data_range(@data); # 5

  $v = $ml->de_mean(1,2,3); # [-1,0,1]

  $y = $ml->variance(@data); # 2.5538

  $y = $ml->standard_deviation(@data); # 1.5981

  $y = $ml->interquartile_range(@data); # 2

  $y = $ml->covariance(\@data, \@data2); # 2.5538

  $y = $ml->correlation(\@data, \@data2); # 0.2474

=head1 METHODS

=head2 mean

  $y = $ml->mean(@data);

=cut

sub mean {
    my ($self, @data) = @_;
    return sum0(@data) / @data;
}

=head2 median

  $y = $ml->median(@data);

=cut

sub median {
    my ($self, @data) = @_;
    @data = sort { $a <=> $b } @data;
    my $result;
    my $high_mid = @data / 2;
    if (@data % 2) {
        $result = $data[$high_mid];
    }
    else {
        $result = ($data[$high_mid - 1] + $data[$high_mid]) / 2;
    }
    return $result;
}

=head2 quantile

  $y = $ml->quantile($percentile, @data);

=cut

sub quantile {
    my ($self, $percentile, @data) = @_;
    my $p = int $percentile * @data;
    @data = sort { $a <=> $b } @data;
    return $data[$p];
}

=head2 mode

  $v = $ml->mode(@data);

=cut

sub mode {
    my ($self, @data) = @_;
    my %seen;
    $seen{$_}++
        for @data;
    my $max = max(values %seen);
    return [ grep { $seen{$_} == $max } sort { $a <=> $b } keys %seen ];
}

=head2 data_range

  $y = $ml->data_range(@data);

=cut

sub data_range {
    my ($self, @data) = @_;
    return max(@data) - min(@data);
}

=head2 de_mean

  $v = $ml->de_mean(@data);

Translate data by sutbtracting its mean from each element, so the result has C<mean=0>.

=cut

sub de_mean {
    my ($self, @data) = @_;
    my $mean = $self->mean(@data);
    return [ map { $_ - $mean } @data ];
}

=head2 variance

  $y = $ml->variance(@data);

=cut

sub variance {
    my ($self, @data) = @_;
    die 'At least 2 data-points reqired'
        unless @data >= 2;
    my $deviations = $self->de_mean(@data);
    return $self->sum_of_squares($deviations) / (@data - 1);
}

=head2 standard_deviation

  $y = $ml->standard_deviation(@data);

=cut

sub standard_deviation {
    my ($self, @data) = @_;
    return sqrt $self->variance(@data);
}

=head2 interquartile_range

  $y = $ml->interquartile_range(@data);

=cut

sub interquartile_range {
    my ($self, @data) = @_;
    return $self->quantile(0.75, @data) - $self->quantile(0.25, @data);
}

=head2 covariance

  $y = $ml->covariance(\@data1, \@data2);

=cut

sub covariance {
    my ($self, $data1, $data2) = @_;
    die 'Data lists must be the same size'
        unless @$data1 == @$data2;
    my $x = $self->de_mean(@$data1);
    my $y = $self->de_mean(@$data2);
    return $self->vector_dot($x, $y) / (@$data1 - 1);
}

=head2 correlation

  $y = $ml->correlation(\@data1, \@data2);

=cut

sub correlation {
    my ($self, $data1, $data2) = @_;
    my $x = $self->standard_deviation(@$data1);
    my $y = $self->standard_deviation(@$data2);
    my $result;
    if ($x > 0 && $y > 0) {
        $result = $self->covariance($data1, $data2) / $x / $y;
    }
    else {
        $result = 0;
    }
    return $result;
}

1;
__END__

=head1 SEE ALSO

L<Data::MachineLearning::Elements>

F<t/002-statistics.t>

L<List::Util>

L<Moo::Role>

=cut
