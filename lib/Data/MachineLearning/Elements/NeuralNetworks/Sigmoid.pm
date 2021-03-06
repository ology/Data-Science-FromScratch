package Data::MachineLearning::Elements::NeuralNetworks::Sigmoid;

use Moo;
use strictures 2;
use namespace::clean;

extends 'Data::MachineLearning::Elements::NeuralNetworks::Layer';

=head1 SYNOPSIS

  use Data::MachineLearning::Elements::NeuralNetworks::Sigmoid;

  my $layer = Data::MachineLearning::Elements::NeuralNetworks::Sigmoid->new;

=head1 DESCRIPTION

A C<Data::MachineLearning::Elements::NeuralNetworks::Sigmoid> is a class for building neural networks.

=head1 ATTRIBUTES

=head2 sigmoids

  $v = $layer->sigmoids;

=cut

has sigmoids => (
    is => 'rw',
);

=head1 METHODS

=head2 forward

  $v = $layer->forward($input);

=cut

sub forward {
    my ($self, $input) = @_;
    my $sigmoids = $self->ml->tensor_apply(sub { $self->ml->sigmoid(shift()) }, $input);
    $self->sigmoids($sigmoids);
    return $self->sigmoids;
}

=head2 backward

  $v = $layer->backward($gradient);

=cut

sub backward {
    my ($self, $gradient) = @_;
    return $self->ml->tensor_combine(
        sub { my ($x, $y) = @_; $x * (1 - $x) * $y },
        $self->sigmoids,
        $gradient
    );
}

1;

=head1 SEE ALSO

L<Data::MachineLearning::Elements::NeuralNetworks::Layer>

L<Moo>

=cut
