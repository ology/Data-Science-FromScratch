package Data::MachineLearning::Elements::NeuralNetworks::Optimizer;

use Data::MachineLearning::Elements;
use Moo;
use strictures 2;
use namespace::clean;

=head1 SYNOPSIS

  # This module is a base class and should not be used directly.

=head1 DESCRIPTION

A C<Data::MachineLearning::Elements::NeuralNetworks::Optimizer> is a class for updating neural networks.

=head1 ATTRIBUTES

=head2 ml

  $obj = $optimizer->ml;

C<Data::MachineLearning::Elements> object

=cut

has ml => (
    is        => 'ro',
    lazy      => 1,
    init_args => undef,
    default   => sub { Data::MachineLearning::Elements->new },
);

=head1 METHODS

=head2 step

  $optimizer->step($layer);

=cut

sub step {
    my ($self, $layer) = @_;
}

1;

=head1 SEE ALSO

F<t/015-deep-learning.t>

L<Data::MachineLearning::Elements::NeuralNetworks::GradientDescent>

L<Data::MachineLearning::Elements::NeuralNetworks::Momentum>

L<Moo>

=cut
