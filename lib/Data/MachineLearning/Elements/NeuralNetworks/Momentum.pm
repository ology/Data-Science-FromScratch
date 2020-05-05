package Data::MachineLearning::Elements::NeuralNetworks::Momentum;

use Moo;
use strictures 2;
use namespace::clean;

extends 'Data::MachineLearning::Elements::NeuralNetworks::Optimizer';

=head1 SYNOPSIS

  use Data::MachineLearning::Elements::NeuralNetworks::Momentum;

  my $optimizer = Data::MachineLearning::Elements::NeuralNetworks::Momentum->new;

=head1 DESCRIPTION

A C<Data::MachineLearning::Elements::NeuralNetworks::Momentum> is an class for updating neural networks.

=head1 ATTRIBUTES

=head2 lr

  $rate = $optimizer->lr;

Learning rate

=cut

has lr => (
    is => 'ro',
);

=head2 mo

  $momentum = $optimizer->mo;

Momentum

=cut

has mo => (
    is => 'ro',
);

=head2 updates

  $avg = $optimizer->updates;

Running average

=cut

has updates => (
    is => 'ro',
);

=head1 METHODS

=head2 step

  $optimizer->step($layer);

=cut

sub step {
    my ($self, $layer) = @_;
    unless ($self->updates) {
        my $updates = [map { $self->ml->zeros_like($_) } @{ $layer->grads }];
        $self->updates($updates);
    }
    for my $i (0 .. @{ $self->updates } - 1) {
        $self->updates->[$i] = $self->ml->tensor_combine(
            sub { my ($x, $y) = @_; return $self->mo * $x + (1 - $self->mo) * $y },
            $self->updates->[$i],
            $layer->grads->[$i]
        );
        $layer->params->[$i] = $self->ml->tensor_combine(
            sub { my ($x, $y) = @_; return $x - $self->lr * $y },
            $layer->params->[$i],
            $self->updates->[$i]
        );
    }
}

1;

=head1 SEE ALSO

L<Data::MachineLearning::Elements::NeuralNetworks::Optimizer>

L<Moo>

=cut
