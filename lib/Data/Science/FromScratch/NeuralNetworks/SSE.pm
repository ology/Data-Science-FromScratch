package Data::Science::FromScratch::NeuralNetworks::SSE;

use Moo;
use strictures 2;
use namespace::clean;

extends 'Data::Science::FromScratch::NeuralNetworks::Loss';

=head1 SYNOPSIS

  use Data::Science::FromScratch::NeuralNetworks::SSE;

  my $sse = Data::Science::FromScratch::NeuralNetworks::SSE->new;

=head1 DESCRIPTION

A C<Data::Science::FromScratch::NeuralNetworks::SSE> is an class for computing loss in neural networks.

=head1 METHODS

=head2 loss

  $x = $sse->loss($predicted, $actual);

=cut

sub loss {
    my ($self, $predicted, $actual) = @_;
    my $squared_errors = $self->ds->tensor_combine(
        sub { ($predicted - $actual) ** 2 },
        $predicted,
        $actual
    );
    return $self->ds->tensor_sum($squared_errors);
}

=head2 gradient

  $v = $sse->gradient($gradient);

=cut

sub gradient {
    my ($self, $predicted, $actual) = @_;
    return $self->ds->tensor_combine(
        sub { 2 * ($predicted - $actual) },
        $predicted,
        $actual
    );
}

1;

=head1 SEE ALSO

L<Data::Science::FromScratch::NeuralNetworks::Loss>

=cut
