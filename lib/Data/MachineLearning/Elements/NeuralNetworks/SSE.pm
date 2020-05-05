package Data::MachineLearning::Elements::NeuralNetworks::SSE;

use Moo;
use strictures 2;
use namespace::clean;

extends 'Data::MachineLearning::Elements::NeuralNetworks::Loss';

=head1 SYNOPSIS

  use Data::MachineLearning::Elements::NeuralNetworks::SSE;

  my $loss = Data::MachineLearning::Elements::NeuralNetworks::SSE->new;

=head1 DESCRIPTION

A C<Data::MachineLearning::Elements::NeuralNetworks::SSE> is an class for computing loss in neural networks.

=head1 METHODS

=head2 loss

  $x = $loss->loss($predicted, $actual);

=cut

sub loss {
    my ($self, $predicted, $actual) = @_;
    my $squared_errors = $self->ml->tensor_combine(
        sub { my ($x, $y) = @_; ($x - $y) ** 2 },
        $predicted,
        $actual
    );
    return $self->ml->tensor_sum($squared_errors);
}

=head2 gradient

  $v = $loss->gradient($gradient);

=cut

sub gradient {
    my ($self, $predicted, $actual) = @_;
    return $self->ml->tensor_combine(
        sub { my ($x, $y) = @_; 2 * ($x - $y) },
        $predicted,
        $actual
    );
}

1;

=head1 SEE ALSO

L<Data::MachineLearning::Elements::NeuralNetworks::Loss>

L<Moo>

=cut
