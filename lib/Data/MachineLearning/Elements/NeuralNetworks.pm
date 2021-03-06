package Data::MachineLearning::Elements::NeuralNetworks;

use List::MoreUtils qw(first_index);
use List::Util qw(max);
use Moo::Role;
use Storable qw(dclone);
use strictures 2;

=head1 SYNOPSIS

  use Data::MachineLearning::Elements;

  my $ml = Data::MachineLearning::Elements->new;

  my $y = $ml->step_function(-1); # 0
  $y = $ml->step_function(1); # 1

  $y = $ml->perceptron_output([2,2], -3, [0,1]); # 0
  $y = $ml->perceptron_output([2,2], -1, [0,1]); # 1
  $y = $ml->perceptron_output([-2], 1, [0]); # 1

  $y = $ml->sigmoid(-5); # 0.0067

  $y = $ml->neuron_output(); #

  my $v = $ml->feed_forward(); #

  $v = $ml->sqerror_gradients(); #

  $v = $ml->fizz_buzz_encode(2); # [1,0,0,0]
  $y = $ml->fizz_buzz_accuracy(); #

  $y = $ml->binary_encode(1); # [1,0,0,0,0,0,0,0,0,0]

  $y = $ml->argmax([-1,10,5,20,-3]); # 3

=head1 METHODS

=head2 step_function

  $y = $ml->step_function($x);

=cut

sub step_function {
    my ($self, $x) = @_;
    return $x >= 0 ? 1 : 0;
}

=head2 perceptron_output

  $y = $ml->perceptron_output($weights, $bias, $u);

=cut

sub perceptron_output {
    my ($self, $weights, $bias, $u) = @_;
    my $calculation = $self->vector_dot($weights, $u) + $bias;
    return $self->step_function($calculation);
}

=head2 sigmoid

  $y = $ml->sigmoid($t);

=cut

sub sigmoid {
    my ($self, $t) = @_;
    return $self->logistic($t);
}

=head2 neuron_output

  $y = $ml->neuron_output($weights, $inputs);

=cut

sub neuron_output {
    my ($self, $weights, $inputs) = @_;
    return $self->sigmoid($self->vector_dot($weights, $inputs));
}

=head2 feed_forward

  $v = $ml->feed_forward($neural_network, $input_vector);

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

=head2 sqerror_gradients

  $y = $ml->sqerror_gradients($network, $input_vector, $target_vector);

=cut

sub sqerror_gradients {
    my ($self, $network, $input_vector, $target_vector) = @_;
    # forward pass
    my ($hidden_outputs, $outputs) = @{ $self->feed_forward($network, $input_vector) };
    # gradients with respect to output neuron pre-activation outputs
    my @output_deltas = map {
        $outputs->[$_] * (1 - $outputs->[$_]) * ($outputs->[$_] - $target_vector->[$_])
    } 0 .. @$outputs - 1;
    # gradients with respect to output neuron weights
    my @output_grads;
    for my $i (0 .. @{ $network->[-1] } - 1) {
        my @grad = map { $output_deltas[$i] * $_ } @$hidden_outputs, 1;
        push @output_grads, \@grad;
    }
    # gradients with respect to hidden neuron pre-activation outputs
    my @hidden_deltas;
    for my $i (0 .. @$hidden_outputs - 1) {
        my $x = $hidden_outputs->[$i] * (1 - $hidden_outputs->[$i])
            * $self->vector_dot(\@output_deltas, [ map { $_->[$i] } @{ $network->[-1] } ]);
        push @hidden_deltas, $x;
    }
    # gradients with respect to hidden neuron weights
    my @hidden_grads;
    for my $i (0 .. @{ $network->[0] } - 1) {
        my @grads = map { $hidden_deltas[$i] * $_ } @$input_vector, 1;
        push @hidden_grads, \@grads;
    }
    return [ \@hidden_grads, \@output_grads ];
}

=head2 fizz_buzz_encode

  $v = $ml->fizz_buzz_encode($x);

=cut

sub fizz_buzz_encode {
    my ($self, $x) = @_;
    if ($x % 15 == 0) {
        return [0,0,0,1];
    }
    elsif ($x % 5 == 0) {
        return [0,0,1,0];
    }
    elsif ($x % 3 == 0) {
        return [0,1,0,0];
    }
    else {
        return [1,0,0,0];
    }
}

=head2 fizz_buzz_accuracy

  $y = $ml->fizz_buzz_accuracy($low, $hi, $net);

=cut

sub fizz_buzz_accuracy {
    my ($self, $low, $hi, $net) = @_;
    my $num_correct = 0;
    for my $i ($low .. $hi - 1) {
        my $x = $self->binary_encode($i);
        my $predicted = $self->argmax($net->forward($x));
        my $actual = $self->argmax($self->fizz_buzz_encode($i));
        if ($predicted == $actual) {
            $num_correct++;
        }
    }
    return $num_correct / ($hi - $low);
}

=head2 binary_encode

  $y = $ml->binary_encode($x);

=cut

sub binary_encode {
    my ($self, $x) = @_;
    my @binary;
    for my $i (0 .. 9) {
        push @binary, $x % 2;
        $x = int($x / 2);
    }
    return \@binary;
}

=head2 argmax

  $y = $ml->argmax($vector);

=cut

sub argmax {
    my ($self, $vector) = @_;
    my $max = max(@$vector);
    return first_index { $_ == $max } @$vector;
}

1;
__END__

=head1 SEE ALSO

L<Data::MachineLearning::Elements>

F<t/014-neural-networks.t>

L<Data::MachineLearning::Elements::NeuralNetworks::Layer>

L<Data::MachineLearning::Elements::NeuralNetworks::Linear>

L<Data::MachineLearning::Elements::NeuralNetworks::Sigmoid>

L<Data::MachineLearning::Elements::NeuralNetworks::Sequential>

L<Data::MachineLearning::Elements::NeuralNetworks::Loss>

L<Data::MachineLearning::Elements::NeuralNetworks::SSE>

L<Data::MachineLearning::Elements::NeuralNetworks::Optimizer>

L<Data::MachineLearning::Elements::NeuralNetworks::GradientDescent>

L<List::Util>

L<List::MoreUtils>

L<Moo::Role>

L<Storable>

=cut
