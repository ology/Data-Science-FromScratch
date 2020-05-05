package Data::MachineLearning::Elements::LogisticRegression;

use List::Util qw(sum0);
use Moo::Role;
use strictures 2;

=head1 SYNOPSIS

  use Data::MachineLearning::Elements;

  my $ds = Data::MachineLearning::Elements->new;

  $y = $ds->logistic(0); # 0.5

  $y = $ds->logistic_prime(0); 0.25

  $y = $ds->negative_log_likelihood(); #

  $v = $ds->negative_log_gradient(); #

=head1 METHODS

=head2 logistic

  $y = $ds->logistic($x);

=cut

sub logistic {
    my ($self, $x) = @_;
    return 1 / (1 + exp(-$x));
}

=head2 logistic_prime

  $y = $ds->logistic_prime($x);

=cut

sub logistic_prime {
    my ($self, $x) = @_;
    $x = $self->logistic($x);
    return $x * (1 - $x);
}

=head2 negative_log_likelihood

  $y = $ds->negative_log_likelihood($u, $x, $beta);

=cut

sub negative_log_likelihood {
    my ($self, $u, $x, $beta) = @_;
    return sum0(map { $self->_negative_log_likelihood($u->[$_], $x->[$_], $beta) } 0 .. @$u - 1);
}

sub _negative_log_likelihood {
    my ($self, $u, $x, $beta) = @_;
    if ($x == 1) {
        return - log($self->logistic($self->vector_dot($u, $beta)))
    }
    else {
        return - log(1 - $self->logistic($self->vector_dot($u, $beta)))
    }
}

=head2 negative_log_gradient

  $v = $ds->negative_log_gradient($u, $x, $beta);

This method is suspect because the results from the book code are different.

=cut

sub negative_log_gradient {
    my ($self, $u, $x, $beta) = @_;
    my @nlg = map { $self->_negative_log_gradient($u->[$_], $x->[$_], $beta) } 0 .. @$u - 1;
    return $self->vector_sum(@nlg);
}

sub _negative_log_partial {
    my ($self, $u, $x, $beta, $j) = @_;
    return -($x - $self->logistic($self->vector_dot($u, $beta))) * $u->[$j];
}

sub _negative_log_gradient {
    my ($self, $u, $x, $beta) = @_;
    return [ map { $self->_negative_log_partial($u, $x, $beta, $_) } 0 .. @$beta - 1 ];
}

1;
__END__

=head1 SEE ALSO

L<Data::MachineLearning::Elements>

F</t/012-logistic-regression.t>

L<List::Util>

L<Moo::Role>

=cut
