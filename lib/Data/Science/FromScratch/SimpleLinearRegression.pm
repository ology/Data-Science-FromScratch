package Data::Science::FromScratch::SimpleLinearRegression;

use List::Util qw(sum);
use Moo::Role;
use strictures 2;
use Statistics::Basic qw(mean);

=head1 SYNOPSIS

  use Data::Science::FromScratch;

  my $ds = Data::Science::FromScratch->new;

  my $x = $ds->lr_predict(0.5, 0.5, 0.5); # 0.75

  $x = $ds->lr_error(0.5, 0.5, 0.5, 0.4); # 0.35

  $x = $ds->sum_of_sqerrors(0.5, 0.5,
    [0.1, 0.5, 0.8], [0.2, 0.4, 0.7]); # 0.285

  my ($alpha, $beta) = $ds->lr_least_squares_fit(
    [0.1, 0.5, 0.8], [0.2, 0.4, 0.7]); # 0.1054, 0.7027

  $x = $ds->total_sum_of_squares([1 .. 10]); # 82.5

  $x = $ds->r_squared(); # 

=head1 METHODS

=head2 lr_predict

  $x = $ds->lr_predict($alpha, $beta, $x_i);

=cut

sub lr_predict {
    my ($self, $alpha, $beta, $x_i) = @_;
    return $beta * $x_i + $alpha;
}

=head2 lr_error

  $x = $ds->lr_error($alpha, $beta, $x_i, $y_i);

Actual value: B<y_i>

=cut

sub lr_error {
    my ($self, $alpha, $beta, $x_i, $y_i) = @_;
    return $self->lr_predict($alpha, $beta, $x_i) - $y_i;
}

=head2 sum_of_sqerrors

  $x = $ds->sum_of_sqerrors($alpha, $beta, $x, $y);

=cut

sub sum_of_sqerrors {
    my ($self, $alpha, $beta, $x, $y) = @_;
    my @errors = map { $self->lr_error($alpha, $beta, $x->[$_], $y->[$_]) ** 2 } 0 .. @$x - 1;
    return sum(@errors);
}

=head2 lr_least_squares_fit

  ($alpha, $beta) = $ds->lr_least_squares_fit($x, $y);

This method is suspect because the python results from the book code are different.

=cut

sub lr_least_squares_fit {
    my ($self, $x, $y) = @_;
    my $beta = $self->correlation($x, $y) * $self->standard_deviation(@$y) / $self->standard_deviation(@$x);
    my $alpha = mean($y) - $beta * mean($x);
    return $alpha, $beta;
}

=head2 total_sum_of_squares

  $x = $ds->total_sum_of_squares($v);

=cut

sub total_sum_of_squares {
    my ($self, $v) = @_;
    return sum(map { $_ ** 2 } @{ $self->de_mean(@$v) });
}

=head2 r_squared

  $x = $ds->r_squared($alpha, $beta, $x, $y);

"Coefficient of determination"

=cut

sub r_squared {
    my ($self, $alpha, $beta, $x, $y) = @_;
    return 1 - ($self->sum_of_sqerrors($alpha, $beta, $x, $y) / $self->total_sum_of_squares($y));
}

1;
__END__

=head1 SEE ALSO

L<Data::Science::FromScratch>

F<t/009-simple-linear-regression.t>

L<List::Util>

L<Moo::Role>

L<Statistics::Basic>

=cut
