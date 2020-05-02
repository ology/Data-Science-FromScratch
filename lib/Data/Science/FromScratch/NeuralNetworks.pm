package Data::Science::FromScratch::NeuralNetworks;

use Moo::Role;
use Storable qw(dclone);
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

  $y = $ds->neuron_output(); #

  my $v = $ds->feed_forward(); #

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
    return $self->logistic($t);
}

=head2 neuron_output

  $y = $ds->neuron_output($weights, $inputs);

=cut

sub neuron_output {
    my ($self, $weights, $inputs) = @_;
    return $self->sigmoid($self->vector_dot($weights, $inputs));
}

=head2 feed_forward

  $v = $ds->feed_forward($neural_network, $input_vector);

=cut

sub feed_forward {
    my ($self, $neural_network, $input_vector) = @_;
    my @outputs;
    my $inputs = dclone $input_vector;
    for my $layer (@$neural_network) {
        my @input_with_bias = ( @$inputs, 1 );
        my @output = map { $self->neuron_output($_, \@input_with_bias) } @$layer;
        push @outputs, \@output;
        $inputs = \@output;
    }
    return \@outputs;
}

1;
__END__

=head1 SEE ALSO

L<Data::Science::FromScratch>

F<t/014-neural-networks.t>

L<Moo::Role>

=cut
