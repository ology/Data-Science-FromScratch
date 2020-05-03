package Data::Science::FromScratch::NeuralNetworks::GradientDescent;

use Moo;
use strictures 2;
use namespace::clean;

extends 'Data::Science::FromScratch::NeuralNetworks::Optimizer';

=head1 SYNOPSIS

  use Data::Science::FromScratch::NeuralNetworks::GradientDescent;

  my $gd = Data::Science::FromScratch::NeuralNetworks::GradientDescent->new;

=head1 DESCRIPTION

A C<Data::Science::FromScratch::NeuralNetworks::GradientDescent> is an class for updating neural networks.

=head1 ATTRIBUTES

=head2 lr

  $rate = $gd->lr;

Learning rate

=cut

has lr => (
    is      => 'ro',
    default => sub { 0.1 },
);

=head1 METHODS

=head2 step

  $gd->step($layer);

=cut

sub step {
    my ($self, $layer) = @_;
    for my $i (0 .. @{ $layer->params } - 1) {
        $layer->params->[$i] = $self->ds->tensor_combine(
            sub { $layer->params->[$i] - $layer->grads->[$i] * $self->lr },
            $layer->params->[$i],
            $layer->grads->[$i]
        );
    }
}

1;

=head1 SEE ALSO

L<Data::Science::FromScratch::NeuralNetworks::Optimizer>

=cut
