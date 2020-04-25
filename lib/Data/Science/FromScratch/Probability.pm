package Data::Science::FromScratch::Probability;

use Math::Erf::Approx 'erf';
use Math::Trig ':pi';
use Moo::Role;
use strictures 2;

=head1 SYNOPSIS

  use Data::Science::FromScratch;

  my $ds = Data::Science::FromScratch->new;

  my $x = $ds->uniform_cdf(0); # 0
  $x = $ds->uniform_cdf(0.5); # 0.5
  $x = $ds->uniform_cdf(1); # 1

  $x = $ds->normal_pdf(-1, 0, 1); # 0.2420
  $x = $ds->normal_pdf(0, 0, 1); # 0.3989
  $x = $ds->normal_pdf(1, 0, 1); # 0.2420

  $x = $ds->normal_cdf(-1, 0, 1); # 0.1589
  $x = $ds->normal_cdf(0, 0, 1); # 0.5
  $x = $ds->normal_cdf(1, 0, 1); # 0.8411

  #$x = inverse_normal_cdf($n, $mu, $sigma, $tolerance); # TODO

  $x = $ds->bernouli_trial(0.5); # 0 or 1

  $x = $ds->binomial(100, 0.5); # Greater than or equal to 0

=head1 METHODS

=head2 uniform_cdf

  $x = $ds->uniform_cdf($n);

"Cumulative distribution function"

=cut

sub uniform_cdf {
    my ($self, $n) = @_;
    return $n < 0 ? 0
         : $n < 1 ? $n
         : 1;
}

=head2 normal_pdf

  $x = $ds->normal_pdf($n, $mu, $sigma);

"Probability density function"

=cut

sub normal_pdf {
    my ($self, $n, $mu, $sigma) = @_;
    $mu    ||= 0;
    $sigma ||= 1;
    my $sqrt_two_pi = sqrt pi2;
    my $result = exp(-($n - $mu) ** 2 / 2 / $sigma ** 2) / ($sqrt_two_pi * $sigma);
    return $result;
}

=head2 normal_cdf

  $x = $ds->normal_cdf($n, $mu, $sigma);

=cut

sub normal_cdf {
    my ($self, $n, $mu, $sigma) = @_;
    $mu //= 0;
    $sigma //= 1;
    return (1 + erf(($n - $mu) / sqrt(2) / $sigma)) / 2;
}

=head2 inverse_normal_cdf

  $x = $ds->inverse_normal_cdf($n, $mu, $sigma, $tolerance);

=cut

sub inverse_normal_cdf {
    my ($self, $n, $mu, $sigma, $tolerance) = @_;
    $mu //= 0;
    $sigma //= 1;
    $tolerance //= 0.00001;
    my $result;
    if ($mu != 0 || $sigma != 1) {
        return $mu + $sigma * $self->inverse_normal_cdf($n, 0, 1, $tolerance);
    }
    my $low_z = -10;
    my $hi_z = 10;
    my $mid_z;
    while ($hi_z - $low_z > $tolerance) {
        $mid_z = ($low_z + $hi_z) / 2;
        my $mid_p = $self->normal_cdf($mid_z);
        if ($mid_p < $n) {
            $low_z = $mid_z;
        }
        else {
            $hi_z = $mid_z;
        }
    }
    return $mid_z;
}

=head2 bernouli_trial

  $x = $ds->bernouli_trial($n);

=cut

sub bernouli_trial {
    my ($self, $n) = @_;
    return rand > $n ? 1 : 0;
}

=head2 binomial

  $x = $ds->binomial($n, $p);

B<$n> C<bernouli_trial>s of B<$p>.

=cut

sub binomial {
    my ($self, $n, $p) = @_;
    my $sum = 0;
    $sum += $self->bernouli_trial($p)
        for 1 .. $n;
    return $sum;
}

1;
__END__

=head1 SEE ALSO

L<Data::Science::FromScratch>

F<t/003-probability.t>

L<Math::Erf::Approx>

L<Math::Trig>

L<Moo::Role>

=cut
