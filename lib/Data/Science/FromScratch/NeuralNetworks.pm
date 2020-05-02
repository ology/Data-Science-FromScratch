package Data::Science::FromScratch::NeuralNetworks;

use Moo::Role;
use strictures 2;

=head1 SYNOPSIS

  use Data::Science::FromScratch;

  my $ds = Data::Science::FromScratch->new;

  my $y = $ds->step_function(-1); # 0
  $y = $ds->step_function(1); # 1

  $y = $ds->perceptron_output([2,2], -3, [0,1]); # 0
  $y = $ds->perceptron_output([2,2], -1, [0,1]); # 1
  $y = $ds->perceptron_output([-2], 1, [0]); # 1

  $y = $ds->sigmoid(-5); # 0.0067

=head1 METHODS

=head2 step_function

  $y = $ds->step_function($x);

=cut

sub step_function {
    my ($self, $x) = @_;
    return $x >= 0 ? 1 : 0;
}

=head2 perceptron_output

  $y = $ds->perceptron_output($weights, $bias, $u);

=cut

sub perceptron_output {
    my ($self, $weights, $bias, $u) = @_;
    my $calculation = $self->vector_dot($weights, $u) + $bias;
    return $self->step_function($calculation);
}

=head2 sigmoid

  $y = $ds->sigmoid($t);

=cut

sub sigmoid {
    my ($self, $t) = @_;
    return 1 / (1 + exp(-$t));
}

1;
__END__

=head1 SEE ALSO

L<Data::Science::FromScratch>

F<t/014-neural-networks.t>

L<Moo::Role>

=cut
