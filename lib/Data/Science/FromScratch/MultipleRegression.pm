package Data::Science::FromScratch::MultipleRegression;

use List::Util qw(sum);
use Moo::Role;
use Statistics::Basic qw(median);
use strictures 2;

=head1 SYNOPSIS

  use Data::Science::FromScratch;

  my $ds = Data::Science::FromScratch->new;

  my $x = $ds->mr_predict([1,2,3], [4,4,4]); # 24

  $x = $ds->mr_error([1,2,3], 30, [4,4,4]); # -6

  $x = $ds->squared_error([1,2,3], 30, [4,4,4]); # 36

  my $v = $ds->sqerror_gradient([1,2,3], 30, [4,4,4]); # [-12, -24, -36]

  $v = $ds->mr_least_squares_fit(); # 

  $x = $ds->multiple_r_squared(); # 

  $v = $ds->bootstrap_sample([1,2,3,4,5]); # [5,2,4,2,5] for example

  $v = $ds->bootstrap_statistic([1,2,3,4,5], 100); # [4,3,5,2,...

  $x = $ds->p_value(30.58, 1.27); # < 0.001

  $x = $ds->ridge_penalty([1,2,3,4,5], 0.5); # 27

  $x = $ds->squared_error_ridge(); # 

  $v = $ds->ridge_penalty_gradient(); # 

  $v = $ds->sqerror_ridge_gradient(); # 

  $v = $ds->least_squares_fit_ridge(); # 

  $x = $ds->lasso_penalty([1,2,3,4,5], 0.5); # 7

=head1 METHODS

=head2 mr_predict

  $x = $ds->mr_predict($v, $beta);

=cut

sub mr_predict {
    my ($self, $v, $beta) = @_;
    return $self->vector_dot($v, $beta);
}

=head2 mr_error

  $x = $ds->mr_error($v, $y, $beta);

=cut

sub mr_error {
    my ($self, $v, $y, $beta) = @_;
    return $self->mr_predict($v, $beta) - $y;
}

=head2 squared_error

  $x = $ds->squared_error($v, $y, $beta);

=cut

sub squared_error {
    my ($self, $v, $y, $beta) = @_;
    return $self->mr_error($v, $y, $beta) ** 2;
}

=head2 sqerror_gradient

  $v = $ds->sqerror_gradient($u, $y, $beta);

=cut

sub sqerror_gradient {
    my ($self, $u, $y, $beta) = @_;
    my $error = $self->mr_error($u, $y, $beta);
    return [ map { 2 * $error * $_ } @$u ];
}

=head2 mr_least_squares_fit

  $v = $ds->mr_least_squares_fit($x, $y, $rate, $num_steps, $batch_size);

"In practice, you wouldn't estimate a linear regression using gradient descent..."

This method is suspect because the python results from the book code are different.

=cut

sub mr_least_squares_fit {
    my ($self, $x, $y, $rate, $num_steps, $batch_size) = @_;
    $rate ||= 0.001;
    $num_steps ||= 1000;
    $batch_size ||= 1;
    my $guess = [ map { rand } @{ $x->[0] } ];
    for my $n (1 .. $num_steps) {
        for (my $i = 0; $i < @$x; $i += $batch_size) {
            # Either increment or use array size for end
            my $end = $i + $batch_size - 1;
            $end = $end < @$x ? $end : @$x - 1;
            my @batch_xs = @$x[$i .. $end];
            my @batch_ys = @$y[$i .. $end];
            my @error_grad = map { $self->sqerror_gradient($batch_xs[$_], $batch_ys[$_], $guess) } 0 .. @batch_xs - 1;
            my $gradient = $self->vector_mean(@error_grad);
            $guess = $self->gradient_step($guess, $gradient, -$rate);
        }
    }
    return $guess;
}

=head2 multiple_r_squared

  $x = $ds->multiple_r_squared($u, $v, $beta);

=cut

sub multiple_r_squared {
    my ($self, $u, $v, $beta) = @_;
    my @squared_errors = map { $self->mr_error($u->[$_], $v->[$_], $beta) ** 2 } 0 .. @$u - 1;
    return 1 - sum(@squared_errors) / $self->total_sum_of_squares($v);
}

=head2 bootstrap_sample

  $v = $ds->bootstrap_sample($u);

=cut

sub bootstrap_sample {
    my ($self, $u) = @_;
    return [ map { $u->[int rand @$u ] } @$u ];
}

=head2 bootstrap_statistic

  $v = $ds->bootstrap_statistic($u, $num_samples);

TODO: Handle additional statistical functions (C<$stats_fn>).

=cut

sub bootstrap_statistic {
    my ($self, $u, $num_samples) = @_;
    return [ map { 0 + median($self->bootstrap_sample($u)) } 1 .. $num_samples ];
}

=head2 p_value

  $x = $ds->p_value($beta_hat_j, $sigma_hat_j);

=cut

sub p_value {
    my ($self, $beta_hat_j, $sigma_hat_j) = @_;
    if ($beta_hat_j > 0) {
        return 2 * (1 - $self->normal_cdf($beta_hat_j / $sigma_hat_j));
    }
    else {
        return 2 * $self->normal_cdf($beta_hat_j / $sigma_hat_j);
    }
}

=head2 ridge_penalty

  $x = $ds->ridge_penalty($u, $alpha);

=cut

sub ridge_penalty {
    my ($self, $u, $alpha) = @_;
    return $alpha * $self->vector_dot([ @$u[1 .. @$u - 1] ], [ @$u[1 .. @$u - 1] ]);
}

=head2 squared_error_ridge

  $x = $ds->squared_error_ridge($u, $v, $beta, $alpha);

=cut

sub squared_error_ridge {
    my ($self, $u, $v, $beta, $alpha) = @_;
    return $self->squared_error($u, $v, $beta) + $self->ridge_penalty($beta, $alpha);
}

=head2 ridge_penalty_gradient

  $v = $ds->ridge_penalty_gradient($u, $alpha);

=cut

sub ridge_penalty_gradient {
    my ($self, $u, $alpha) = @_;
    return [ 0, map { 2 * $alpha * $_ } @$u[1 .. @$u - 1] ];
}

=head2 sqerror_ridge_gradient

  $v = $ds->sqerror_ridge_gradient($u, $y, $beta, $alpha);

=cut

sub sqerror_ridge_gradient {
    my ($self, $u, $y, $beta, $alpha) = @_;
    return $self->vector_sum(
        $self->sqerror_gradient($u, $y, $beta),
        $self->ridge_penalty_gradient($beta, $alpha)
    );
}

=head2 least_squares_fit_ridge

  $v = $ds->least_squares_fit_ridge($x, $y, $alpha, $rate, $num_steps, $batch_size);

This method is suspect because the python results from the book code are different.

=cut

sub least_squares_fit_ridge {
    my ($self, $x, $y, $alpha, $rate, $num_steps, $batch_size) = @_;
    $rate ||= 0.001;
    $num_steps ||= 1000;
    $batch_size ||= 1;
    my $guess = [ map { rand } @{ $x->[0] } ];
    for my $n (1 .. $num_steps) {
        for (my $i = 0; $i < @$x; $i += $batch_size) {
            # Either increment or use array size for end
            my $end = $i + $batch_size - 1;
            $end = $end < @$x ? $end : @$x - 1;
            my @batch_xs = @$x[$i .. $end];
            my @batch_ys = @$y[$i .. $end];
            my @error_grad = map { $self->sqerror_ridge_gradient($batch_xs[$_], $batch_ys[$_], $guess, $alpha) } 0 .. @batch_xs - 1;
            my $gradient = $self->vector_mean(@error_grad);
            $guess = $self->gradient_step($guess, $gradient, -$rate);
        }
    }
    return $guess;
}

=head2 lasso_penalty

  $x = $ds->lasso_penalty($u, $alpha);

=cut

sub lasso_penalty {
    my ($self, $u, $alpha) = @_;
    return $alpha * sum(map { abs $_ } @$u[1 .. @$u - 1]);
}

1;
__END__

=head1 SEE ALSO

L<Data::Science::FromScratch>

F<t/010-multiple-regression.t>

L<List::Util>

L<Moo::Role>

=cut
