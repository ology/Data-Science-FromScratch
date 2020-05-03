package Data::Science::FromScratch::NeuralNetworks::Loss;

use Data::Science::FromScratch;
use Moo;
use strictures 2;
use namespace::clean;

=head1 SYNOPSIS

  # This module is a base class and should not be used directly.

=head1 DESCRIPTION

A C<Data::Science::FromScratch::NeuralNetworks::Loss> is an abstract class for computing loss in neural networks.

=head1 ATTRIBUTES

=head2 ds

  $obj = $loss->ds;

C<Data::Science::FromScratch> object

=cut

has ds => (
    is        => 'ro',
    lazy      => 1,
    init_args => undef,
    default   => sub { Data::Science::FromScratch->new },
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

L<Data::Science::FromScratch::NeuralNetworks::SSE>

=cut
