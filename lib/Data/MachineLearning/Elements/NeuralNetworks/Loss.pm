package Data::MachineLearning::Elements::NeuralNetworks::Loss;

use Data::MachineLearning::Elements;
use Moo;
use strictures 2;
use namespace::clean;

=head1 SYNOPSIS

  # This module is a base class and should not be used directly.

=head1 DESCRIPTION

A C<Data::MachineLearning::Elements::NeuralNetworks::Loss> is an abstract class for computing loss in neural networks.

=head1 ATTRIBUTES

=head2 ml

  $obj = $loss->ml;

C<Data::MachineLearning::Elements> object

=cut

has ml => (
    is        => 'ro',
    lazy      => 1,
    init_args => undef,
    default   => sub { Data::MachineLearning::Elements->new },
);

=head1 METHODS

=head2 loss

  $x = $loss->loss($predicted, $actual);

=cut

sub loss {
    my ($self, $predicted, $actual) = @_;
}

=head2 gradient

  $v = $loss->gradient($gradient);

=cut

sub gradient {
    my ($self, $predicted, $actual) = @_;
}

1;

=head1 SEE ALSO

F<t/015-deep-learning.t>

L<Data::MachineLearning::Elements::NeuralNetworks::SSE>

L<Moo>

=cut
