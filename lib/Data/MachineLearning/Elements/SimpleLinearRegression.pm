package Data::MachineLearning::Elements::SimpleLinearRegression;

use List::Util qw(sum0);
use Moo::Role;
use strictures 2;
use Statistics::Basic qw(mean);

=head1 SYNOPSIS

  use Data::MachineLearning::Elements;

  my $ml = Data::MachineLearning::Elements->new;

  my $y = $ml->lr_predict(0.5, 0.5, 0.5); # 0.75

  $y = $ml->lr_error(0.5, 0.5, 0.5, 0.4); # 0.35

  $y = $ml->sum_of_sqerrors(0.5, 0.5,
    [0.1, 0.5, 0.8], [0.2, 0.4, 0.7]); # 0.285

  my ($alpha, $beta) = $ml->lr_least_squares_fit(
    [0.1, 0.5, 0.8], [0.2, 0.4, 0.7]); # 0.1054, 0.7027

  $y = $ml->total_sum_of_squares([1 .. 10]); # 82.5

  $y = $ml->r_squared(); #

=head1 METHODS

=head2 lr_predict

  $y = $ml->lr_predict($alpha, $beta, $x);

=cut

sub lr_predict {
    my ($self, $alpha, $beta, $x) = @_;
    return $beta * $x + $alpha;
}

=head2 lr_error

  $z = $ml->lr_error($alpha, $beta, $x, $y);

Actual value: B<y_i>

=cut

sub lr_error {
    my ($self, $alpha, $beta, $x, $y) = @_;
    return $self->lr_predict($alpha, $beta, $x) - $y;
}

=head2 sum_of_sqerrors

  $z = $ml->sum_of_sqerrors($alpha, $beta, $x, $y);

=cut

sub sum_of_sqerrors {
    my ($self, $alpha, $beta, $x, $y) = @_;
    my @errors = map { $self->lr_error($alpha, $beta, $x->[$_], $y->[$_]) ** 2 } 0 .. @$x - 1;
    return sum0(@errors);
}

=head2 lr_least_squares_fit

  ($alpha, $beta) = $ml->lr_least_squares_fit($x, $y);

=cut

sub lr_least_squares_fit {
    my ($self, $x, $y) = @_;
    my $beta = $self->correlation($x, $y) * $self->standard_deviation(@$y) / $self->standard_deviation(@$x);
    my $alpha = mean($y) - $beta * mean($x);
    return $alpha, $beta;
}

=head2 total_sum_of_squares

  $y = $ml->total_sum_of_squares($v);

=cut

sub total_sum_of_squares {
    my ($self, $v) = @_;
    return sum0(map { $_ ** 2 } @{ $self->de_mean(@$v) });
}

=head2 r_squared

  $z = $ml->r_squared($alpha, $beta, $x, $y);

"Coefficient of determination"

=cut

sub r_squared {
    my ($self, $alpha, $beta, $x, $y) = @_;
    return 1 - ($self->sum_of_sqerrors($alpha, $beta, $x, $y) / $self->total_sum_of_squares($y));
}

1;
__END__

=head1 SEE ALSO

L<Data::MachineLearning::Elements>

F<t/010-simple-linear-regression.t>

L<List::Util>

L<Moo::Role>

L<Statistics::Basic>

=cut
