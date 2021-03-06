package Data::MachineLearning::Elements::Inference;

use Moo::Role;
use strictures 2;

=head1 SYNOPSIS

  use Data::MachineLearning::Elements;

  my $ml = Data::MachineLearning::Elements->new;

  my $v = $ml->normal_approximation_to_binomial(100, 0.5); # [50,5]

  my $y = $ml->normal_probability_below(-1, 0, 1); # 0.1589
  $y = $ml->normal_probability_above(-1, 0, 1); # 0.8411
  $y = $ml->normal_probability_between(-1, 0, 0, 1); # 0.3411
  $y = $ml->normal_probability_outside(-1, 0, 0, 1); # 0.6589

  $y = $ml->normal_upper_bound(0.05, 0, 1); # -1.6431
  $y = $ml->normal_lower_bound(0.05, 0, 1); # 1.6431
  $v = $ml->normal_two_sided_bounds(0.05, 0, 1); # [-0.0632, 0.0632]

  $y = $ml->two_sided_p_value(1, 0, 1); # 0.3178

=head1 METHODS

=head2 normal_approximation_to_binomial

  $y = $ml->normal_approximation_to_binomial($n, $p);

=cut

sub normal_approximation_to_binomial {
    my ($self, $n, $p) = @_;
    my $mu = $p * $n;
    my $sigma = sqrt($p * (1 - $p) * $n);
    return [ $mu, $sigma ];
}

=head2 normal_probability_below

  $y = $ml->normal_probability_below($lo, $mu, $sigma);

=cut

sub normal_probability_below {
    my $self = shift;
    $self->normal_cdf(@_);
}

=head2 normal_probability_above

  $y = $ml->normal_probability_above($lo, $mu, $sigma);

=cut

sub normal_probability_above {
    my ($self, $lo, $mu, $sigma) = @_;
    $mu //= 0;
    $sigma //= 1;
    return 1 - $self->normal_cdf($lo, $mu, $sigma);
}

=head2 normal_probability_between

  $y = $ml->normal_probability_between($lo, $hi, $mu, $sigma);

=cut

sub normal_probability_between {
    my ($self, $lo, $hi, $mu, $sigma) = @_;
    $mu //= 0;
    $sigma //= 1;
    return $self->normal_cdf($hi, $mu, $sigma) - $self->normal_cdf($lo, $mu, $sigma);
}

=head2 normal_probability_outside

  $y = $ml->normal_probability_outside($lo, $hi, $mu, $sigma);

=cut

sub normal_probability_outside {
    my ($self, $lo, $hi, $mu, $sigma) = @_;
    $mu //= 0;
    $sigma //= 1;
    return 1 - $self->normal_probability_between($lo, $hi, $mu, $sigma);
}

=head2 normal_upper_bound

  $y = $ml->normal_upper_bound($probability, $mu, $sigma);

=cut

sub normal_upper_bound {
    my ($self, $probability, $mu, $sigma) = @_;
    $mu //= 0;
    $sigma //= 1;
    return $self->inverse_normal_cdf($probability, $mu, $sigma);
}

=head2 normal_lower_bound

  $y = $ml->normal_lower_bound($probability, $mu, $sigma);

=cut

sub normal_lower_bound {
    my ($self, $probability, $mu, $sigma) = @_;
    $mu //= 0;
    $sigma //= 1;
    return $self->inverse_normal_cdf(1 - $probability, $mu, $sigma);
}

=head2 normal_two_sided_bounds

  $v = $ml->normal_two_sided_bounds($probability, $mu, $sigma);

=cut

sub normal_two_sided_bounds {
    my ($self, $probability, $mu, $sigma) = @_;
    $mu //= 0;
    $sigma //= 1;
    my $tail_probability = (1 - $probability) / 2;
    my $upper_bound = $self->normal_lower_bound($tail_probability, $mu, $sigma);
    my $lower_bound = $self->normal_upper_bound($tail_probability, $mu, $sigma);
    return [$lower_bound, $upper_bound];
}

=head2 two_sided_p_value

  $v = $ml->two_sided_p_value($n, $mu, $sigma);

=cut

sub two_sided_p_value {
    my ($self, $n, $mu, $sigma) = @_;
    $mu //= 0;
    $sigma //= 1;
    my $result;
    if ($n >= $mu) {
        $result = 2 * $self->normal_probability_above($n, $mu, $sigma);
    }
    else {
        $result = 2 * $self->normal_probability_below($n, $mu, $sigma);
    }
    return $result;
}

1;
__END__

=head1 SEE ALSO

L<Data::MachineLearning::Elements>

F<t/004-inference.t>

L<Moo::Role>

=cut
