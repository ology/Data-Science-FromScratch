package Data::MachineLearning::Elements::NeuralNetworks::Layer;

use Data::MachineLearning::Elements;
use Moo;
use strictures 2;
use namespace::clean;

=head1 SYNOPSIS

  # This module is a base class and should not be used directly.

=head1 DESCRIPTION

A C<Data::MachineLearning::Elements::NeuralNetworks::Layer> is an abstract class for building neural networks.

=head1 ATTRIBUTES

=head2 ds

  $obj = $layer->ds;

C<Data::MachineLearning::Elements> object

=cut

has ds => (
    is        => 'ro',
    lazy      => 1,
    init_args => undef,
    default   => sub { Data::MachineLearning::Elements->new },
);

=head1 METHODS

=head2 forward

  $v = $layer->forward($input);

=cut

sub forward {
    my ($self, $input) = @_;
}

=head2 backward

  $v = $layer->backward($gradient);

=cut

sub backward {
    my ($self, $gradient) = @_;
}

=head2 params

  $v = $layer->params;

=cut

sub params {
    my ($self) = @_;
    return [];
}

=head2 grads

  $v = $layer->grads;

=cut

sub grads {
    my ($self) = @_;
    return [];
}

1;

=head1 SEE ALSO

F<t/015-deep-learning.t>

L<Data::MachineLearning::Elements::NeuralNetworks::Sigmoid>

L<Data::MachineLearning::Elements::NeuralNetworks::Linear>

L<Data::MachineLearning::Elements::NeuralNetworks::Sequential>

L<Moo>

=cut
