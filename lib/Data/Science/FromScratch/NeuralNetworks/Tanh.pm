package Data::Science::FromScratch::NeuralNetworks::Tanh;

use Moo;
use strictures 2;
use namespace::clean;

extends 'Data::Science::FromScratch::NeuralNetworks::Layer';

=head1 SYNOPSIS

  use Data::Science::FromScratch::NeuralNetworks::Tanh;

  my $layer = Data::Science::FromScratch::NeuralNetworks::Tanh->new;

=head1 DESCRIPTION

A C<Data::Science::FromScratch::NeuralNetworks::Tanh> is a class for building neural networks.

=head1 ATTRIBUTES

=head2 tanh

  $v = $layer->tanh;

=cut

has tanh => (
    is => 'rw',
);

=head1 METHODS

=head2 forward

  $v = $layer->forward($input);

=cut

sub forward {
    my ($self, $input) = @_;
    my $tanh = $self->ds->tensor_apply(sub { $self->ds->tanh(shift()) }, $input);
    $self->tanh($tanh);
    return $self->tanh;
}

=head2 backward

  $v = $layer->backward($gradient);

=cut

sub backward {
    my ($self, $gradient) = @_;
    return $self->ds->tensor_combine(
        sub { my ($x, $y) = @_; (1 - $x ** 2) * $y },
        $self->tanh,
        $gradient
    );
}

1;

=head1 SEE ALSO

L<Data::Science::FromScratch::NeuralNetworks::Layer>

=cut
