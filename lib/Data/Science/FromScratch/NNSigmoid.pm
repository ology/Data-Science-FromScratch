package Data::Science::FromScratch::NNSigmoid;

use Moo;
use strictures 2;
use namespace::clean;

extends 'Data::Science::FromScratch::NNLayer';

=head1 SYNOPSIS

  use Data::Science::FromScratch::Sigmoid;

  my $sigmoid = Data::Science::FromScratch::Sigmoid->new;

=head1 DESCRIPTION

A C<Data::Science::FromScratch::Sigmoid> is a class for building neural networks.

=head1 ATTRIBUTES

=head2 sigmoids

  $v = $sigmoid->sigmoids;

=cut

has sigmoids => (
    is => 'rw',
);

=head1 METHODS

=head2 forward

  $v = $sigmoid->forward($input);

=cut

sub forward {
    my ($self, $input) = @_;
    my $sigmoids = $self->ds->tensor_apply(sub { return $self->ds->sigmoid(shift()) }, $input);
    $self->sigmoids($sigmoids);
    return $self->sigmoids;
}

=head2 backward

  $v = $sigmoid->backward($gradient);

=cut

sub backward {
    my ($self, $gradient) = @_;
    return $self->ds->tensor_combine(
        sub { my ($sig, $grad) = @_; return $sig * (1 - $sig) * $grad },
        $self->sigmoids,
        $gradient
    );
}

1;