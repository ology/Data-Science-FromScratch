package Data::Science::FromScratch::NeuralNetworks::Optimizer;

use Data::Science::FromScratch;
use Moo;
use strictures 2;
use namespace::clean;

=head1 SYNOPSIS

  # This module is a base class and should not be used directly.

=head1 DESCRIPTION

A C<Data::Science::FromScratch::NeuralNetworks::Optimizer> is a class for updating neural networks.

=head1 ATTRIBUTES

=head2 ds

  $obj = $optimizer->ds;

C<Data::Science::FromScratch> object

=cut

has ds => (
    is        => 'ro',
    lazy      => 1,
    init_args => undef,
    default   => sub { Data::Science::FromScratch->new },
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

L<Data::Science::FromScratch::NeuralNetworks::GradientDescent>

L<Data::Science::FromScratch::NeuralNetworks::Momentum>

L<Moo>

=cut
