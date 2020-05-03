package Data::Science::FromScratch::NNGradientDescent;

use Moo;
use strictures 2;
use namespace::clean;

extends 'Data::Science::FromScratch::NNOptimizer';

=head1 SYNOPSIS

  use Data::Science::FromScratch::NNGradientDescent;

  my $gd = Data::Science::FromScratch::NNGradientDescent->new;

=head1 DESCRIPTION

A C<Data::Science::FromScratch::NNGradientDescent> is an class for updating neural networks.

=head1 ATTRIBUTES

=head2 lr

  $rate = $gd->lr;

Learning rate

=cut

has lr => (
    is => 'ro',
);

=head1 METHODS

=head2 step

  $x = $gd->step($layer);

=cut

sub step {
    my ($self, $layer) = @_;
    for my $i (0 .. @{ $layer->params } - 1) {
        $layer->params->[$i] = $self->ds->tensor_combine(
            sub { my ($x, $y) = @_; return $x - $y * $self->lr },
            $layer->params->[$i],
            $layer->grads->[$i]
        );
    }
}

1;

=head1 SEE ALSO

L<Data::Science::FromScratch::NNOptimizer>

=cut
