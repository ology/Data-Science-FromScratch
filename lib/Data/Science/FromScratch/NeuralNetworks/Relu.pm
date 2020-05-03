package Data::Science::FromScratch::NeuralNetworks::Relu;

use List::Util qw(max);
use Moo;
use strictures 2;
use namespace::clean;

extends 'Data::Science::FromScratch::NeuralNetworks::Layer';

=head1 SYNOPSIS

  use Data::Science::FromScratch::NeuralNetworks::Relu;

  my $layer = Data::Science::FromScratch::NeuralNetworks::Relu->new;

=head1 DESCRIPTION

A C<Data::Science::FromScratch::NeuralNetworks::Relu> is a class for building neural networks.

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

L<Data::Science::FromScratch::NeuralNetworks::Layer>

L<List::Util>

L<Moo>

=cut
