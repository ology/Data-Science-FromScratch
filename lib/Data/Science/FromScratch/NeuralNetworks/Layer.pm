package Data::Science::FromScratch::NeuralNetworks::Layer;

use Data::Science::FromScratch;
use Moo;
use strictures 2;
use namespace::clean;

=head1 SYNOPSIS

  # This module is a base class and should not be used directly.

=head1 DESCRIPTION

A C<Data::Science::FromScratch::NeuralNetworks::Layer> is an abstract class for building neural networks.

=head1 ATTRIBUTES

=head2 ds

  $obj = $layer->ds;

C<Data::Science::FromScratch> object

=cut

has ds => (
    is        => 'ro',
    lazy      => 1,
    init_args => undef,
    default   => sub { Data::Science::FromScratch->new },
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
    return ();
}

=head2 grads

  $v = $layer->grads;

=cut

sub grads {
    my ($self) = @_;
    return ();
}

1;

=head1 SEE ALSO

F<t/015-deep-learning.t>

L<Data::Science::FromScratch::NeuralNetworks::Sigmoid>

L<Data::Science::FromScratch::NeuralNetworks::Linear>

L<Data::Science::FromScratch::NeuralNetworks::Sequential>

=cut
