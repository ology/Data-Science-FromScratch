package Data::Science::FromScratch::LogisticRegression;

use List::Util qw(sum0);
use Moo::Role;
use strictures 2;

=head1 SYNOPSIS

  use Data::Science::FromScratch;

  my $ds = Data::Science::FromScratch->new;

  $x = $ds->logistic(0); # 0.5

  $x = $ds->logistic_prime(0); 0.25

  $x = $ds->negative_log_likelihood(); #

  $v = $ds->negative_log_gradient(); #

=head1 METHODS

=head2 logistic

  $x = $ds->logistic($y);

=cut

sub logistic {
    my ($self, $y) = @_;
    return 1 / (1 + exp(-$y));
}

=head2 logistic_prime

  $x = $ds->logistic_prime($y);

=cut

sub logistic_prime {
    my ($self, $y) = @_;
    $y = $self->logistic($y);
    return $y * (1 - $y);
}

=head2 negative_log_likelihood

  $x = $ds->negative_log_likelihood($u, $y, $beta);

=cut

sub negative_log_likelihood {
    my ($self, $u, $y, $beta) = @_;
    return sum0(map { $self->_negative_log_likelihood($u->[$_], $y->[$_], $beta) } 0 .. @$u - 1);
}

sub _negative_log_likelihood {
    my ($self, $u, $y, $beta) = @_;
    if ($y == 1) {
        return - log($self->logistic($self->vector_dot($u, $beta)))
    }
    else {
        return - log(1 - $self->logistic($self->vector_dot($u, $beta)))
    }
}

=head2 negative_log_gradient

  $v = $ds->negative_log_gradient($u, $y, $beta);

=cut

sub negative_log_gradient {
    my ($self, $u, $y, $beta) = @_;
    my @nlg = map { $self->_negative_log_gradient($u->[$_], $y->[$_], $beta) } 0 .. @$u - 1;
    return $self->vector_sum(@nlg);
}

sub _negative_log_partial {
    my ($self, $u, $y, $beta, $j) = @_;
    return -($y - $self->logistic($self->vector_dot($u, $beta))) * $u->[$j];
}

sub _negative_log_gradient {
    my ($self, $u, $y, $beta) = @_;
    return [ map { $self->_negative_log_partial($u, $y, $beta, $_) } 0 .. @$beta - 1 ];
}

1;
__END__

=head1 SEE ALSO

L<Data::Science::FromScratch>

F</t/012-logistic-regression.t>

L<List::Util>

L<Moo::Role>

=cut
