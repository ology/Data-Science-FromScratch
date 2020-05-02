package Data::Science::FromScratch::NNSequential;

use Moo;
use strictures 2;
use namespace::clean;

=head1 SYNOPSIS

  use Data::Science::FromScratch::NNSequential;

  my $seq = Data::Science::FromScratch::NNSequential->new;

=head1 DESCRIPTION

A C<Data::Science::FromScratch::NNSequential> is a class for building neural networks.

=head1 ATTRIBUTES

=head2 layers

  $obj = $seq->layers;

=cut

has layers => (
    is => 'rw',
);

=head1 METHODS

=head2 forward

  $v = $layer->forward($input);

=cut

sub forward {
    my ($self, $input) = @_;
}

=head2 backward

  $v = $layer->backward($gradient);

=cut

sub backward {
    my ($self, $gradient) = @_;
}

=head2 params

  $v = $layer->params;

=cut

sub params {
    my ($self) = @_;
    return ();
}

=head2 grads

  $v = $layer->grads;

=cut

sub grads {
    my ($self) = @_;
    return ();
}

=head1 SEE ALSO

L<Data::Science::FromScratch::NNLayer>

1;
