package Data::MachineLearning::Elements::NeuralNetworks::Relu;

use List::Util qw(max);
use Moo;
use strictures 2;
use namespace::clean;

extends 'Data::MachineLearning::Elements::NeuralNetworks::Layer';

=head1 SYNOPSIS

  use Data::MachineLearning::Elements::NeuralNetworks::Relu;

  my $layer = Data::MachineLearning::Elements::NeuralNetworks::Relu->new;

=head1 DESCRIPTION

A C<Data::MachineLearning::Elements::NeuralNetworks::Relu> is a class for building neural networks.

=head1 ATTRIBUTES

=head2 input

  $v = $layer->input;

=cut

has input => (
    is => 'rw',
);

=head1 METHODS

=head2 forward

  $v = $layer->forward($input);

=cut

sub forward {
    my ($self, $input) = @_;
    $self->input($input);
    return $self->ds->tensor_apply(sub { max(shift(), 0) }, $input);
}

=head2 backward

  $v = $layer->backward($gradient);

=cut

sub backward {
    my ($self, $gradient) = @_;
    return $self->ds->tensor_combine(
        sub { my ($x, $y) = @_; $x > 0 ? $y : 0 },
        $self->input,
        $gradient
    );
}

1;

=head1 SEE ALSO

L<Data::MachineLearning::Elements::NeuralNetworks::Layer>

L<List::Util>

L<Moo>

=cut
