package Data::MachineLearning::Elements::NeuralNetworks::GradientDescent;

use Moo;
use strictures 2;
use namespace::clean;

extends 'Data::MachineLearning::Elements::NeuralNetworks::Optimizer';

=head1 SYNOPSIS

  use Data::MachineLearning::Elements::NeuralNetworks::GradientDescent;

  my $optimizer = Data::MachineLearning::Elements::NeuralNetworks::GradientDescent->new;

=head1 DESCRIPTION

A C<Data::MachineLearning::Elements::NeuralNetworks::GradientDescent> is an class for updating neural networks.

=head1 ATTRIBUTES

=head2 lr

  $rate = $optimizer->lr;

Learning rate

=cut

has lr => (
    is      => 'ro',
    default => sub { 0.1 },
);

=head1 METHODS

=head2 step

  $optimizer->step($layer);

=cut

sub step {
    my ($self, $layer) = @_;
    for my $i (0 .. @{ $layer->params } - 1) {
        $layer->params(
            $i,
            $self->ml->tensor_combine(
                sub { my ($x, $y) = @_; $x - $y * $self->lr },
                $layer->params->[$i],
                $layer->grads->[$i]
            )
        );
    }
}

1;

=head1 SEE ALSO

L<Data::MachineLearning::Elements::NeuralNetworks::Optimizer>

L<Moo>

=cut
